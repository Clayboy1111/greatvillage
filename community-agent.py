from dotenv import load_dotenv
from twilio.rest import Client
import twilio_keys
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

# Code for loading document and creating VectorStore
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)

pdf_reader = PdfReader("./data/w_brain_doc_v2.pdf")
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

chunks = text_splitter.split_text(text=text)

embeddings = OpenAIEmbeddings()
VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

# Twilio setup
client = Client(twilio_keys.account_sid, twilio_keys.auth_token)

# Langchain setup
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


# Tool for similarity search
@tool
def find_similarity(incoming_prompt):
    """
    Find similarity in the VectorStore based on the incoming prompt.

    Parameters:
    - incoming_prompt (str): The input prompt.

    Returns:
    - List[Document]: List of documents from the VectorStore.
    """
    return VectorStore.similarity_search(query=incoming_prompt)


tools = [find_similarity]

# ChatPromptTemplate for Langchain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Your name is Hatch and you are an expert on the operations 
            and culture of Hatch Towers. questions from residents in a professional, fun, and responsible way"""
            """r""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="Hatch"),
    ]
)

# Binding tools to llm
llm_with_tools = llm.bind_tools(tools)

# Creating the agent
agent = (
    {
        "input": lambda x: x["input"],
        "Hatch": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Agent Executor
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

# Flask App
app = Flask(__name__)
CORS(app)


@app.route("/sms", methods=["POST"])
def sms_reply():
    # add access list
    incoming_number = request.values.get("From").strip()
    accesslist = twilio_keys.w300_access_list
    give_access = len(accesslist) > 0 and incoming_number in accesslist

    reply = MessagingResponse()
    if not give_access:
        reply.message(
            """Sorry, but u don't have access to the Hatch Towers community😞. If you want to be a resident at Hatch Towers we can connect you with our partner real estate agency OR you can just give Nick Petkoff a Lamborghini. Hope to speak 2u soon 👋 """
        )
        return str(reply)

    incoming_prompt = request.values.get("Body").strip()
    # docs = VectorStore.similarity_search(query=incoming_prompt)
    response = agent_executor.invoke({"input": incoming_prompt})

    str_response = response["output"]

    reply = MessagingResponse()
    reply.message(str_response)
    return str(reply)


if __name__ == "__main__":
    app.run(debug=True, port=3702)
