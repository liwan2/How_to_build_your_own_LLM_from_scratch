import json
import os
import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

#liwan_config
class liwan_config(PretrainedConfig):
    model_name="liwan"
    def __init__(self,hidden_size:int=768,num_hidden_layer:int=8,use_mox:bool=False,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size=hidden_size
        self.num_hidden_layer=num_hidden_layer
        self.use_mox=use_mox
        self.num_attention_heads=kwargs.get("num_attention_heads",8)
        self.num_key_value_heads=kwargs.get("num_key_value_heads",4)
        self.head_dim=kwargs.get("head_dim",self.hidden_size//self.num_attention_heads)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.dropout = kwargs.get("dropout", 0.0)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        #MOE相关参数
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)#权重归一化开关
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.vocab_size = kwargs.get("vocab_size", 8000)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)#推理阶段外推才使用
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

#liwan_model

class RMSnorm(nn.Module):
    """RMSNorm实现"""
    def __init__(self,dim:int,eps:float=1e-8):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def RMS(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        return self.RMS(x.float())*self.weight.type_as(x)
    
def precompute_freqs_cis(dim,end:int=int(32*1024),rope_base:float=1e6,rope_scale:dict=None)->tuple[torch.Tensor, torch.Tensor]:
    """提前计算位置编码的频率"""
    # self.rope_scaling = {
    #         "beta_fast": 32,
    #         "beta_slow": 1,
    #         "factor": 16,
    #         "original_max_position_embeddings": 2048,
    #         "attention_factor": 1.0,
    #         "type": "yarn"
    #     } if self.inference_rope_scaling else None
    freqs,atten_factor=1.0/(rope_base**((torch.arange(0,dim,2))[0:dim//2].float()/dim)),1.0
    if rope_scale is not None:
        beta_fast,beta_slow,factor,ori_max,atten_factor=(
            rope_scale.get("beta_fast", 32),rope_scale.get("beta_slow", 1),rope_scale.get("factor", 16),
            rope_scale.get("original_max_position_embeddings", 2048),rope_scale.get("attention_factor", 1.0))
        if end/ori_max>1.0:
            inv_dim=lambda b:(dim*math.log(ori_max/(b*2*math.pi)))/(2*math.log(rope_base))
            low,high=max(0,math.floor(inv_dim(beta_fast))),min(dim//2-1,math.ceil(inv_dim(beta_slow)))
            ramp=torch.clamp((torch.arange(dim//2,device=freqs.device).float()-low)/max(high-low,0.01),0,1)
            freqs=freqs*(1-ramp+ramp/factor)

    t=torch.arange(end,device=freqs.device)
    freqs=torch.outer(t,freqs).float()
    freqs_cos=torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)*atten_factor
    freqs_sin=torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)*atten_factor
    return freqs_cos,freqs_sin

def rotate_half_simple(x, cos, sin):
    """旋转一半的向量"""
    d=x.shape[-1]
    x1=x[...,:d//2]
    x2=x[...,d//2:]
    x11=x1*cos[...,:d//2]-x2*sin[...,:d//2]
    x12=x1*sin[...,:d//2]+x2*cos[...,:d//2]
    return torch.cat([x11, x12], dim=-1)

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    """应用旋转位置编码"""
    cos=cos.unsqueeze(unsqueeze_dim)
    sin=sin.unsqueeze(unsqueeze_dim)
    q_embed = rotate_half_simple(q, cos, sin).to(q.dtype)
    k_embed = rotate_half_simple(k, cos, sin).to(k.dtype)
    return q_embed, k_embed
def repeat_kv(x,n_rep):
    """复制kv以便满足q的数量"""
    bs,slen,num_key_value,dim=x.shape
    if n_rep==1:return x
    x=x[:,:,:,None,:].expand(bs,slen,num_key_value,n_rep,dim)
    x=x.reshape(bs,slen,num_key_value*n_rep,dim)
    return x

class Attention(nn.Module):
    """注意力机制实现"""
    def __init__(self,config:liwan_config):
        super().__init__()
        self.num_key_value_heads=config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.num_attention_heads=config.num_attention_heads
        self.n_local_heads=self.num_attention_heads
        self.n_local_key_value_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_key_value_heads
        self.head_dim=config.head_dim
        self.hidden_size=config.hidden_size
        self.is_causal=True
        self.q_proj=nn.Linear(self.hidden_size,self.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        #GQA,MQA中，q与k,v的维度不一致，q的维度为num_attention_heads*head_dim，k与v的维度为num_key_value_heads*head_dim
        self.o_proj=nn.Linear(self.num_attention_heads*self.head_dim,self.hidden_size,bias=False)#把多头注意力结果拼回去
        self.q_norm=RMSnorm(self.head_dim,config.rms_norm_eps)
        self.k_norm=RMSnorm(self.head_dim,config.rms_norm_eps)
        self.attn_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.dropout=config.dropout
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and config.flash_attn
    
    def forward(self,x,pos_embedding,use_cache=False,past_key_value=None,attention_mask=None):
        cos,sin=pos_embedding
        bsz,seq_len,_=x.shape
        xq=self.q_proj(x)
        xk=self.k_proj(x)
        xv=self.v_proj(x)
        xq=xq.view(bsz,seq_len,self.num_attention_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xq=self.q_norm(xq)
        xk=self.k_norm(xk)
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        present_key_value=(xk,xv) if use_cache else None
        xk,xv=repeat_kv(xk,self.n_rep),repeat_kv(xv,self.n_rep)
        xq,xk,xv=xq.transpose(1,2),xk.transpose(1,2),xv.transpose(1,2)
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output=F.scaled_dot_product_attention(xq,xk,xv,dropout_p=self.dropout if self.training else 0.0,is_causal=self.is_causal)
        else:
            scores=xq@xk.transpose(-2,-1)/math.sqrt(self.head_dim)
            matrix=torch.full((seq_len,seq_len),float("-inf"),device=scores.device).triu(1)
            if self.is_causal:
                scores+=matrix
            if attention_mask is not None:
                scores+=(1.0-attention_mask.unsqueeze(1).unsqueeze(2))*(-1e9)
            output=self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.o_proj(output)
        output=self.resid_dropout(output)
        return output,present_key_value
    # (bsz, seq_len, hidden)
    # → 拆多头 → (bsz, seq_len, heads, head_dim)
    # → KV缓存拼接 → K/V 变长
    # → 复制KV头 → K/V头数 = Q头数
    # → 转置 → (bsz, heads, seq_len, head_dim)
    # → Q@K.T → (bsz, heads, seq_len, past_len+seq_len)
    # → softmax @ V → (bsz, heads, seq_len, head_dim)
    # → 转置+reshape → (bsz, seq_len, hidden)

class Feedback(nn.Module):
    """前馈网络实现"""
    def __init__(self,config:liwan_config,intermediate_size=None):
        super().__init__()
        self.hidden_size=config.hidden_size
        self.intermediate_size=intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
        self.act_fn=ACT2FN[config.hidden_act]

    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
    
class MOEFeedback(nn.Module):
    """MOE前馈网络实现"""
    def __init__(self,config:liwan_config):
        super().__init__()
        self.config=config
        self.gate=nn.Linear(config.hidden_size,config.num_experts,bias=False)
        self.experts=nn.ModuleList([Feedback(config,config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn=ACT2FN[config.hidden_act]

    def forward(self,x):
        """每个 token → 选 topk 专家 → 只丢给对应专家算 → 加权加回 → 顺便均衡专家负载"""
        bsz,seq_len,dim=x.shape
        x_flat=x.view(-1,dim)
        scores=self.gate(x_flat)
        scores=F.softmax(scores.float(),dim=-1)
        topk_weight,topk_idx=torch.topk(scores,self.config.num_experts_per_tok,dim=-1,sorted=False)
        if self.config.norm_topk_prob:
            topk_weight=topk_weight/(topk_weight.sum(dim=-1,keepdim=True)+1e-20)
        y=torch.zeros_like(x_flat)
        for i,expert in enumerate(self.experts):
            mask=(topk_idx==i)
            if mask.any():
                token_idx=mask.any(dim=-1).nonzero().flatten()
                weight=topk_weight[mask].view(-1,1)
                y.index_add_(0,token_idx,(expert(x_flat[token_idx])*weight)).to(y.dtype)
            elif self.training:
                y[0,0]+=0*sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef>0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().sum(dim=1).mean(0)  
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(bsz, seq_len, dim)
    
class liwanBlock(nn.Module):
    """liwan模型的基本模块"""
    def __init__(self, config:liwan_config,layer_id:int):
        super().__init__()
        self.config=config
        self.layer_num=layer_id
        self.self_attn=Attention(config)
        self.input_layernorm=RMSnorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm=RMSnorm(config.hidden_size,config.rms_norm_eps)
        self.MLP=MOEFeedback(config) if config.use_mox else Feedback(config)

    def forward(self,hidden_status,pos_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual=hidden_status
        hidden_status=self.input_layernorm(hidden_status)
        hidden_status,present_key_value=self.self_attn(hidden_status,pos_embeddings,past_key_value,use_cache,attention_mask)
        hidden_status+=residual
        hidden_status=hidden_status+self.MLP(self.post_attention_layernorm(hidden_status))
        return hidden_status,present_key_value
    
class liwanModel(nn.Module):
    def __init__(self,config:liwan_config):
        super().__init__()
        self.config=config
        self.num_hidden_layer,self.vocab_size=self.config.num_hidden_layer,self.config.vocab_size
        self.embeddings=nn.Embedding(self.vocab_size,self.config.hidden_size)
        self.dropout=nn.Dropout(self.config.dropout)
        self.layers=nn.ModuleList([liwanBlock(self.config,i) for i in range(self.num_hidden_layer)])
        self.norm=RMSnorm(self.config.hidden_size,self.config.rms_norm_eps)
        freqs_cos,freqs_sin=precompute_freqs_cis(self.config.head_dim,end=self.config.max_position_embeddings,rope_base=1e6,rope_scale=None)
        self.freqs_cos:torch.Tensor
        self.freqs_sin:torch.Tensor
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
        batch_size,seq_len=input_ids.shape
        if hasattr(past_key_values,'layers'):past_key_values=None
        past_key_values=past_key_values or [None]*len(self.layers)
        start_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_status=self.dropout(self.embeddings(input_ids))
        """判断正余弦表是否初始化，否则重新运算"""
        if self.freqs_cos[0, 0] == 0:
            freqs_cos,freqs_sin=precompute_freqs_cis(self.config.head_dim,end=self.config.max_position_embeddings,rope_base=1e6,rope_scale=None)
            freqs_cos,freqs_sin=freqs_cos.to(hidden_status.device),freqs_sin.to(hidden_status.device)
        pos_embeddings=(self.freqs_cos[start_pos:start_pos+seq_len,:],self.freqs_sin[start_pos:start_pos+seq_len,:])
        presents=[]
        for layer,past_key_value in zip(self.layers,past_key_values):
            hidden_status,present=layer(hidden_status,pos_embeddings,past_key_value,past_key_value,use_cache,attention_mask)
            presents.append(present)
        hidden_status=self.norm(hidden_status)
        aux_loss=sum(getattr(layer.MLP,'aux_loss',0) for layer in self.layers)#getattr(obj, 'attr', default)→ 有 attr 就返回它，没有就返回 0
        return hidden_status,presents,aux_loss

class liwanForcasualLLM(PreTrainedModel,GenerationMixin):
    config_class=liwan_config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self,config:liwan_config=None):
        self.config=config or liwan_config()
        super().__init__(config)
        self.model=liwanModel(self.config)
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        if self.config.tie_word_embeddings:
            self.model.embeddings.weight=self.lm_head.weight
        self.post_init()

    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False,logits_to_keep=0,labels=None,**kwargs):
        hidden_status,past_key_values,aux_loss=self.model(input_ids,attention_mask,past_key_values,use_cache,**kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits=self.lm_head(hidden_status[:,slice_indices,:])
        loss=None
        if labels is not None:
            x,y=logits[...,:-1,:].contiguous(),labels[...,1:].contiguous()
            loss=F.cross_entropy(x.view(-1,self.vocab_size),y.view(-1),ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss,aux_loss=aux_loss,logits=logits,past_key_values=past_key_values,hidden_states=hidden_status)
    
    @torch.inference_mode()
    def generate(self,inputs=None,attention_mask=None,max_new_tokens=8192,temperature=0.85,top_p=0.85,top_k=50,eos_token_id=2,streamer=None,use_cache=True,num_return_sequences=1,do_sample=True,repetition_penalty=1.0,**kwargs):
        input_ids=kwargs.pop("input_ids",inputs).repeat(num_return_sequences,1)
        attention_mask=attention_mask.repeat(num_return_sequences,1) if attention_mask is not None else None
        past_key_value=kwargs.pop("past_key_value",None)
        finished=torch.zeros(input_ids[0],dtype=torch.bool,device=input_ids.device)
        if streamer:streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len=past_key_value[0][0].shape[1] if past_key_value else 0#是seq_len
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_value, use_cache=use_cache, **kwargs)
            if attention_mask is not None:
                new_mask = attention_mask.new_ones(attention_mask.shape[0], 1)
                attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            else:
                attention_mask = None
            logits=outputs.logits[:,-1,:]/temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen=torch.unique(input_ids[i])
                    score=logits[i,seen]
                    logits[i,seen]=torch.where(score<0,score*repetition_penalty,score/repetition_penalty)
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                threshold = topk_vals[..., -1, None]  
                logits[logits < threshold] = -float('inf')
            if top_p<1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)#计算累计概率
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False  # 第一个永远不屏蔽
                original_mask = torch.zeros_like(mask)
                original_mask.scatter_(dim=-1, index=sorted_indices, src=mask)
                logits[original_mask] = -float('inf')
            next_token=torch.multinomial(torch.softmax(logits,dim=-1),num_samples=1) if do_sample else torch.argmax(logits,dim=-1,keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )
            input_ids=torch.cat([input_ids,next_token],dim=-1)
            past_key_value=outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_value}
        return input_ids


        












        
