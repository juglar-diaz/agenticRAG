from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
import requests
import json
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

from src.prompts import RAG_PROMPT_TEMPLATE, ROUTER_AGENT_PROMPT_TEMPLATE

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT", "")  # @param {type:"string"}
LOCATION = os.getenv("GCP_LOCATION", "")  # @param {type:"string"}

if PROJECT_ID == "":
    print("Warning: GCP_PROJECT is not set")

if LOCATION == "":
    print("Warning: GCP_LOCATION is not set")


import vertexai
#vertexai.init(project=PROJECT_ID, location=LOCATION)
vertexai.init()

# Setup the models
# Configure embedding model, for example "text-embedding-004".
embed_model = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

model = "gemini-1.5-pro-001"
llm = VertexAI(model_name=model)

# Load the documents
ai_history_docs = PyPDFLoader("data/ai_history.pdf").load_and_split()

# Split the documents
text_splitter_ai_history = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
doc_splits_ai_history = text_splitter_ai_history.split_documents(ai_history_docs)

vectorstore_ai_history = Chroma.from_documents(documents=doc_splits_ai_history, embedding=embed_model)

# Setup the retriever
retriever_ai_history = vectorstore_ai_history.as_retriever(search_type="similarity", search_kwargs={"k": 1})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["question", "context"]
)

response_chain = (rag_prompt
    | llm
    | StrOutputParser()

)


router_prompt = PromptTemplate(
    template=ROUTER_AGENT_PROMPT_TEMPLATE, input_variables=["question"]
)


router_chain = router_prompt | llm | JsonOutputParser()

########### Create Nodes Actions ###########
class AgentState(TypedDict):
    question : str
    answer : str
    documents : List[str]

def route_question(state):
    """
    Route question to agents to retrieve relevant data

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTING---")
    question = state["question"]
    result = router_chain.invoke({"question": question})
    return result["agent"]

def retrieve_ai_history(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever_ai_history.invoke(question)
    return {"documents": documents, "question": question}

def retrieve_albums(state):
    """
    Retrieve info from API and Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved information from API
    """
    print("---USE API---")

    response_API = requests.get('https://jsonplaceholder.typicode.com/albums')
    # print(response_API.status_code)
    list_albums = json.loads(response_API.text)
    documents = [Document(album['title']) for album in list_albums]
    documents = [Document('music albums give sense to live')]
    question = state["question"]
    return {"documents": documents, "question": question}

def retrieve_albums_ai(state):
    """
    Retrieve documents from API

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved information from API
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever_ai_history.invoke(question)

    print("---USE API---")

    response_API = requests.get('https://jsonplaceholder.typicode.com/albums')
    # print(response_API.status_code)
    list_albums = json.loads(response_API.text)

    api_documents = [Document(album['title']) for album in list_albums]

    api_documents = [Document('music albums give sense to live')]#This example is to pass info that makes sense to the LLM.

    documents = list(documents) + api_documents
    question = state["question"]
    print(documents)

    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using retrieved data

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    answer = response_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "answer": answer}

def start(state):
    """
     Answer when to msg is unrelated to the data

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    answer = "Sorry, I only can help you by answering questions about AI and albums."
    return {"question": question, "answer": answer}


########### Build Execution Graph ###########
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("retrieve_albums", retrieve_albums)
workflow.add_node("retrieve_ai_history", retrieve_ai_history)
workflow.add_node("retrieve_albums_ai", retrieve_albums_ai)
workflow.add_node("generate", generate)
workflow.add_node("start", start)

workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve_albums": "retrieve_albums",
        "retrieve_ai_history": "retrieve_ai_history",
        "retrieve_albums_ai": "retrieve_albums_ai",
        "start": "start",
    },
)

workflow.add_edge("retrieve_ai_history", "generate")
workflow.add_edge("retrieve_albums", "generate")
workflow.add_edge("retrieve_albums_ai", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("start", END)


app = workflow.compile()
