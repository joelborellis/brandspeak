import os
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from dotenv import load_dotenv
from azure.search.documents.models import (
    VectorizedQuery,
    VectorFilterMode,    
)

load_dotenv()

class Search:
    
    def __init__(self):
        # assign the Search variables for Azure Cogintive Search - use .env file and in the web app configure the application settings
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
        credential_search = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL")

        self.sc = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=credential_search)
        self.model = OPENAI_EMBED_MODEL
        self.client = OpenAI()
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def search_hybrid(self, query: str) -> str:
        print(f"Starting Search - {query}")
        vector_query = VectorizedQuery(vector=self.get_embedding(query), k_nearest_neighbors=5, fields="contentVector")
        print("Got embedding")
        results = []
        
        r = self.sc.search(  
            search_text=None,  
            vector_queries= [vector_query],
            vector_filter_mode=VectorFilterMode.PRE_FILTER,
            select=["category", "sourcefile", "content"],
        )
        print("Got result")
        for doc in r:
                results.append(f"[SOURCEFILE:  {doc['sourcefile']}]" + doc['content'])
        print("\n".join(results))
        return ("\n".join(results))
    
    def search_hybrid(self, query: str, category: str) -> str:
        print(f"Starting Search - {query}")
        vector_query = VectorizedQuery(vector=self.get_embedding(query), k_nearest_neighbors=2, fields="contentVector")
        print("Got embedding")
        results = []
        
        r = self.sc.search(  
            search_text=None,  
            vector_queries= [vector_query],
            vector_filter_mode=VectorFilterMode.PRE_FILTER,
            filter=f"category eq '{category}'",
            select=["category", "sourcefile", "content"],
        )
        print("Got result")
        for doc in r:
                results.append(f"[SOURCEFILE:  {doc['sourcefile']}]" + doc['content'])
        print("\n".join(results))
        return ("\n".join(results))