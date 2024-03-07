import pickle
import json
import argparse
from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
import requests
import re
import structlog
import spacy

logger = structlog.get_logger(__name__)

#nlp = spacy.load("en_core_web_sm")


class DataLoader:
    def __init__(self, url, create_docs=True, chunk_size=4000, chunk_overlap=50):
        self.url = url
        self.create_retriever = create_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_or_load_retriever(self):
        if self.create_retriever:
            return self.create_retriever_instance()  # Call the function to create a retriever
        else:
            return self.load_retriever("docs.pickle")

    def replace_newline_with_space(self, text):
        return text.replace("\n", " ")

    def preprocess_text(self, text):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.is_stop != True]
        text = " ".join(tokens)
        text = re.sub(r'\s+', ' ', text)
        return text

    def create_retriever_instance(self):
        text = self.load_text_from_url()
        #text = self.preprocess_text(text)  
        # chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        text_splitter = CharacterTextSplitter() #includes heuristics for non 
        texts = text_splitter.split_text(text)
        #print("text", type(texts[0]), len(texts), texts[0], list(map(len,texts)))
        # docs_metadata = [{"id": f"doc_{i}"} for i in range(len(texts))]

        # docs = text_splitter.create_documents(texts,docs_metadata)
        # print("docs after split",type(docs[0]), len(docs),docs[0])
        
        docs = []
        for idx, text in enumerate(texts):
            text_metadata = {'page_content': text, 'metadata': { 'source':idx}}
            docs.append(Document(**text_metadata))

        # docs = []
        # for chunk in chunk_dicts:
        #     doc = Document(page_content = str(chunk['page_content'])[13:], metadata = chunk['metadata'])
        #     docs.append(doc)
        
        # print(len(docs))
        # self.save_retriever(docs, 'docs.json')
        #self.save_retriever(docs, 'docs.pickle')
        return docs

    def save_retriever(self, docs, filename):
        with open(filename, 'wb') as json_file:
            docs_ = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            #json.dump(docs_, json_file)
            #pickle.dump(docs_, json_file)
        logger.debug(f"Retriever saved to {filename}")

    @classmethod
    def from_dict(cls, data):
        page_content = data.get('page_content')
        metadata = data.get('metadata')
        return cls(page_content, metadata)

    def load_retriever(self, filename):
        with open(filename, 'rb') as file:
            json_string = pickle.load(file)
            documents = self.json_to_documents(json_string)
        return documents
    
    def load_text_from_url(self):
        response = requests.get(self.url)
        return response.text

    def json_to_documents(self, json_string):
        documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json_string]
        return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create or load a retriever object.')
    parser.add_argument('--create_retriever', type=bool, default=True, help='Whether to create a new retriever (True) or load from existing retriever.pkl (False)')
    args = parser.parse_args()

    url = 'https://classics.mit.edu/Homer/iliad.mb.txt'

    dataloader = DataLoader(url, create_retriever=args.create_retriever)
    logger.debug(dataloader.create_retriever)

    if dataloader.create_retriever:
        retriever = dataloader.create_or_load_retriever(dataloader.create_retriever)
        logger.debug("[main.py ] retriever " + str(retriever))


