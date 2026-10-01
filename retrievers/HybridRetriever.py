from typing import List, Tuple
import os
import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from retrievers.Retriever import Retriever
from retrievers.Bm25 import Bm25
from retrievers.Sbert import Sbert
from retrievers.Qwen import Qwen
from retrievers.BGE import Bge
from retrievers.Gte import Gte
from retrievers.Nomic import Nomic
from retrievers.Random import Random
from retrievers.Fuzzy import Fuzzy
from retrievers.Tfidf import Tfidf
from retrievers.Item import Item

from rankers.Ranker import Ranker
from rankers.TfidfRanker import TfidfRanker
from rankers.CrossEncoderModel import CrossEncoderModel
from rankers.BGEReranker import BGEReranker
from rankers.SbertRanker import SbertRanker

def get_retriever(retriever_name: str, catalogue: str, output_length: int) -> Retriever:
    if retriever_name == 'bm25':
        return Bm25(data_source=catalogue, output_length=output_length)
    elif retriever_name == 'sbert_512':
        return Sbert(data_source=catalogue, output_length=output_length, size=512)
    elif retriever_name == 'sbert_768':
        return Sbert(data_source=catalogue, output_length=output_length, size=768)
    elif retriever_name == 'sbert_1024':
        return Sbert(data_source=catalogue, output_length=output_length, size=1024)
    elif retriever_name == 'qwen_1024':
        return Qwen(data_source=catalogue, output_length=output_length, size=1024)
    elif retriever_name == 'qwen_2560':
        return Qwen(data_source=catalogue, output_length=output_length, size=2560)
    elif retriever_name == 'qwen_4096':
        return Qwen(data_source=catalogue, output_length=output_length, size=4096)
    elif retriever_name == 'bge_dense':
        return Bge(data_source=catalogue, output_length=output_length, return_dense=True, return_sparse=False)
    elif retriever_name == 'bge_sparse':
        return Bge(data_source=catalogue, output_length=output_length, return_dense=False, return_sparse=True)
    elif retriever_name == 'gte':
        return Gte(data_source=catalogue, output_length=output_length)
    elif retriever_name == 'nomic':
        return Nomic(data_source=catalogue, output_length=output_length)
    elif retriever_name == 'random':
        return Random(data_source=catalogue, output_length=output_length, random_seed=123)
    elif retriever_name == 'fuzzy_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='ratio')
    elif retriever_name == 'fuzzy_sort_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_sort_ratio')
    elif retriever_name == 'fuzzy_set_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_set_ratio')
    elif retriever_name == 'tfidf':
        return Tfidf(data_source=catalogue, output_length=output_length)
    else:
        raise ValueError(f"Unknown model: {retriever_name}")

def get_ranker(ranker_name: str, output_length: int) -> Ranker:
    if ranker_name == "tfidf_ranker":
        return TfidfRanker(output_length)
    elif ranker_name == "sbert_1024_ranker":
        return SbertRanker(output_length, size=1024)
    else:
        raise ValueError(f"Unknown ranker: {ranker_name}")

class HybridRetriever(Retriever):
    def __init__(self, data_source: str, output_length: int, retriever_name:str, retriever_length:int, ranker_name:str):
        super().__init__(data_source, output_length)
        self.retriever = get_retriever(retriever_name, data_source, retriever_length)
        self.retriever_length = retriever_length
        self.ranker = get_ranker(ranker_name, output_length)
        
    def retrieve(self, query, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()

        # retrieve items using the specified retriever
        retrieved_items, retrieval_time = self.retriever.retrieve(query, language=language)

        print(len(retrieved_items), "items retrieved in", retrieval_time, "seconds")

        # rank the retrieved items using the specified ranker
        ranked_items = self.ranker.rank(retrieved_items, query, language=language)

        return ranked_items, time.time() - start
