#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain


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


#Chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True
)

chunks = text_splitter.split_documents(pages)


# Salvar no Vector DB = Chroma

db = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="text_index_2")


#Carregar DB

vectordb = Chroma(persist_directory="text_index_2", embedding_function=embedding_model)

#Load Retriever
retriever = vectordb.as_retriever(search_kwargs={"k":3})

#Construção da cadeia de promt para chamda do LLM
chain = load_qa_chain(llm, chain_type="stuff")


def ask(question):
    context = retriever.get_relevant_documents(question)
    answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']
    return answer


user_question = input("User: ")
answer = ask(user_question)
print("Answer: ", answer)

