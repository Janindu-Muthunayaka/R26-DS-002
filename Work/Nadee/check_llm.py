from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")
print(llm.invoke("ආයුබෝවන්! ඔබට සිංහල තේරෙනවාද?").content)