import time
import re
from rouge_score import rouge_scorer
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
import openai
import os
import structlog

logger = structlog.get_logger(__name__)

API_TOKEN = os.environ.get('OPENAI_API_KEY')

def format_qa_answer(result):
        question = result['question']
        answer = result['answer']
        sources = result['sources']
        source_documents = result['source_documents']
        sources = []

        # Deduplicate source documents and convert to list of Documents
        source_docs = set()
        for doc in source_documents:
            source_docs.add(doc.page_content)
            sources.append(str(doc.metadata))
        source_docs = [Document(page_content=doc) for doc in source_docs]

        return question, answer, sources, source_docs 


def function_qa_with_sources(query, docs):
    
    output = {'answer': "", 'sources':"", 'source_documents':""}

    # Check if query is non-empty string
    if not isinstance(query, str) or not query.strip():
        print("Invalid query. Please enter a non-empty string.")
        return output

    # Check if query contains alphanumeric 
    if not re.search("^[a-zA-Z0-9]", query):
        print("Invalid query. Please enter a string containing alphanumeric characters.")
        return output

    embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2") 
    
    # classic vectorstore approach 
    chroma_db = Chroma.from_documents(docs, embedding_function ) # persist_directory="./chroma_db") 

    # dense retrieval technique
    #retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", 
    #                        search_kwargs={"score_threshold": 0.1, "k": 10})
    retriever = chroma_db.as_retriever()
    retriever.get_relevant_documents(query)  # euclidean distance, could use cosine similarity instead



    openai.api_key = API_TOKEN
    llm = ChatOpenAI(
            openai_api_key=openai.api_key,
            model_name='gpt-3.5-turbo',
            temperature=0.4,
            #max_tokens=500,
            verbose = True,
            max_retries=2
    )
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff", #faster, the 'refine' option is more elaborate
            retriever=retriever,
            return_source_documents = True,
            #max_tokens_limit = 2000,
            #reduce_k_below_max_tokens = True
            )

    output = qa_with_sources(query)
    print(output)
    question, answer, sources, source_docs = format_qa_answer(output)
        
    return {'question' :question ,'answer': answer, 'sources':set(sources), 'source_documents':source_docs}


def evaluate_rag_model(query, result, expected, docs):
    from nltk.translate.bleu_score import SmoothingFunction

    # Measure query latency
    start_time = time.time()
    result = function_qa_with_sources(query, docs)
    end_time = time.time()
    query_latency = end_time - start_time

    # Extract generated answer from the result
    generated_answer = result['answer']

    # bleu score not overall sens - rouge score better, but still sensible to wording
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 
    scores = scorer.score(generated_answer, expected)
    rouge_score = scores['rougeL'].fmeasure
    
    return {'query_latency': query_latency, 'rouge_score': rouge_score} 
