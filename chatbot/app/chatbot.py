import json
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from langchain.chains import RetrievalQA
import time
import logging
import re
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from pinecone import Pinecone, ServerlessSpec
import pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_key = os.environ.get("PINECONE_KEY", "default_endpoint")
openai_api_key= os.environ.get("OPENAI_API_KEY", "default_api_key")
#date = time.strftime("%Y-%m-%d %H:%M:%S")
#dynamodb_client = boto3.resource('dynamodb', region_name='us-east-2')   
#table = dynamodb_client.Table('maintenance-rag-bot')

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        question = body['question']
    except:
        question = event['question']
    if not question:
        return {
            'statusCode': 400,
            'body': json.dumps('No question provided')
        }
    
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

    index_name = "aerocivildocs"
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    text_field = "text"
    vectorstore = PineconeVectorStore(
        index, embeddings, text_field, pinecone_api_key=pinecone_key, namespace="aero"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  

    
    retrieved_docs = vectorstore.similarity_search_with_score(question, k=3)
    document_list = [item[0] for item in retrieved_docs]
    retrieved_docs_context = '---\n'.join(f"{doc[0].page_content}\nSource: {(doc[0].metadata)}" for doc in retrieved_docs)

    extracted_data = []
    for doc in document_list:
        title = doc.metadata.get('title', 'N/A')
        extracted_data.append(title)
        answer_aditionaldata = extracted_data
    
    prompt_template = """
    Como asistente experto, mis respuestas se basan únicamente en los documentos recuperados de nuestra base de datos de proyectos. Proporcionaré una respuesta detallada y profunda a la pregunta utilizando estos documentos como contexto. También citaré todos los documentos fuentes proporcionando el documento del título que respalda mi respuesta  y la página en la que se encuentra. Sigue las siguientes instrucciones:
    1. Todas las respuestas que voy a proporcionar serán únicamente en Español.
    2- Si no tengo contexto para dar una respuesta voy a responder: "No tengo suficiente contexto para proporcionar una respuesta adecuada"
    Este es el formato que seguiré:
    Respuesta: [Mi respuesta basada en el contexto]
    Todos los dcumentos de referencia: titulo del documento1 y número de página, titulo del documento2 y número de página, ...
    Pregunta: {question}
    Contexto: {context}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 20000, 'temperature': 0.3, 'stop_sequences': ['Human:']})
    gpt = OpenAI(temperature=0.3, model_name='gpt-4-0613',max_tokens=20000)
    try:
        chain = RetrievalQA.from_chain_type(llm=gpt, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
        complete_query = f"Question: {question}\n\nContext:\n{retrieved_docs_context}"
        answer_llm = chain.run({"query": complete_query})
    except:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
        complete_query = f"Question: {question}\n\nContext:\n{retrieved_docs_context}"
        answer_llm = chain.run({"query": complete_query})
    #table.put_item(Item={'date': date, 'question': question, 'answer': answer_llm})
    
    return {
        'respuesta':json.dumps(answer_llm),
        'fuente': json.dumps(answer_aditionaldata)
        }