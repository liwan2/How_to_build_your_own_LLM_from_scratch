from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="gpt-4", 
    temperature=0.7, 
    api_key="your_api_key_here"
    )
