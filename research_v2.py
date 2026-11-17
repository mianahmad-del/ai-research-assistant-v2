from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage  # ← NEW
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
load_dotenv(override=True)


llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

search_tool = TavilySearch(max_results = 3)

vectorstore = Chroma(
    persist_directory="./research_db",
    embedding_function=embeddings
)

summary = ""
recent = []
messages_count=0

def search_store(query):
    print("searching websites")

    results = search_tool.invoke(query)
    docs = []

    for r in results["results"]:
        doc = Document(
            page_content=r["content"],
            metadata={
                "url":r["url"],
                "query":query
            }
        )
        docs.append(doc)

    vectorstore.add_documents(docs)
    print(f"{len(docs)} results save")

    output=""
    for r in results["results"]:
        output += f"URL: {r['url']}\n"
        output += f"Content: {r['content'][:200]}\n\n"

    return output

def smart_search(query):
    print("Checking in Chroma_DB first")

    existing = vectorstore.similarity_search(
        query,
        k=2
    )
    if len(query.split()) < 3:
              return ""
    
    if existing:
        print("Found in Chroma_db, no web search")
        output=""
        for doc in existing:
            output +=f"Content: {doc.page_content[:200]}\n\n"
        return output
       
    else:
        print("Not Found - Searching web")
        return search_store(query)

research_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are the research assistant.
    Use this context to answer:
    {context}

    Conversation summary:
    {summary}
    """),
    MessagesPlaceholder(variable_name="recent"),
    ("user","{question}")
])

research_chain = research_prompt|llm|StrOutputParser()

def get_answer(question):
    global recent,messages_count,summary

    context = smart_search(question)

    response = research_chain.invoke({
       "question": question,
        "context": context,
        "summary": summary,
        "recent": recent       
    })

    recent.append(HumanMessage(content=question))
    recent.append(AIMessage(content=response))

    if len(recent)>4:
        recent = recent[-4:]

    messages_count +=1
    if messages_count % 2 == 0:
        summary += f"\nUser: {question} | Bot: {response[:100]}"
        print(f"📝 Summary updated!")
    return response

print("🚀 Research Assistant v0.2 Ready!")
print("Type 'exit' to quit\n")

while True:
    question = input("You: ")

    if question == "exit":
        print("Good Bye!")
        break

    answer = get_answer(question)
    print(f"\n AI: {answer}")










                     