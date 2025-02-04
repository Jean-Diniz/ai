#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


import os
os.environ["GROQ_API_KEY"] = ""

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1.0,
    max_retries=2,
)


#Carregar pdf

pdf_link = "lotus.pdf"

loader = PyPDFLoader(pdf_link, extract_images=False)
pages = loader.load_and_split()


#Splitters

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True
)


# Storages
store = InMemoryStore()
vector_store = Chroma(persist_directory="child_vector_db", embedding_function=embedding_model)


parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

parent_document_retriever.add_documents(pages, ids=None)


TEMPLATE = """
    Você é um especialista em inteligência artificial. Responda a pergunta abaixo utilizando o contexto informado.
    Query:
    {question}

    Context:
    {context}
"""

rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)


setup_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": parent_document_retriever})

output_parser = StrOutputParser()


parent_chain_retriever = setup_retrieval | rag_prompt | llm | output_parser


parent_chain_retriever.invoke("Porque o TAG com LOTUS é superior ao text2sql e rag?")

