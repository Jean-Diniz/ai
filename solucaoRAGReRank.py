#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank


import os
os.environ["GROQ_API_KEY"] = ""

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=2,
)


#Carregar pdf

pdf_link = "lotus.pdf"

loader = PyPDFLoader(pdf_link, extract_images=False)
pages = loader.load_and_split()


#Separar em chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True
)

chunks = text_splitter.split_documents(pages)
print(chunks)


# Storages
db = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="naiveDB2")

vector_db = Chroma(persist_directory="naiveDB2", embedding_function=embedding_model)


naive_retreiver = vector_db.as_retriever(search_kwargs={"k":10})
print(vector_db.get())

os.environ["COHERE_API_KEY"] = ""

rerank = CohereRerank(top_n=3, model="rerank-v3.5")

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,
    base_retriever=naive_retreiver
)


TEMPLATE = """
    Você é um especialista em inteligência artificial. Responda a pergunta abaixo utilizando o contexto informado.
    Query:
    {question}

    Context:
    {context}
"""

rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)


setup_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": compressor_retriever})

output_parser = StrOutputParser()


compressor_chain_retriever = setup_retrieval | rag_prompt | llm | output_parser


compressor_chain_retriever.invoke("Faça um resumo do contexto?")

