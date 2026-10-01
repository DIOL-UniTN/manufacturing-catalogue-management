from typing import List, Tuple, Dict
from retrievers.Item import Item
from rankers.Ranker import Ranker
from sentence_transformers import SentenceTransformer
import re
import torch

class SbertRanker(Ranker):
    def __init__(self, output_length: int, size: int):
        super().__init__(output_length)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.size = size

        # init model
        if self.size == 512:
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=device)
        elif self.size == 768:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        elif self.size == 1024:
            self.model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
        else:
            raise ValueError(f"Unknown model size: {size}")

    def rank(self, documents: List[Tuple[Item, float]], query: str, language: str) -> List[Tuple[Item, float]]:
        super().rank(documents, query, language)

        # read the documents and populate python dictionary of items
        self.items: Dict[str, Tuple[Item, float]] = {}
        self.catalog_ita: List[str] = []
        self.catalog_eng: List[str] = []
        for item, score in documents:
            self.items[item.item_id] = (item, score)

            if self.language == 'ita':
                text = item.ita_short_desc + ' ' + (item.ita_long_desc or '')
                self.catalog_ita.append(text)
            elif self.language == 'eng':
                text = item.eng_short_desc + ' ' + (item.eng_long_desc or '')
                self.catalog_eng.append(text)
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        
        # create embeddings for both Italian and English catalogs
        if self.language == 'ita':
            embeddings = self.model.encode(self.catalog_ita, convert_to_tensor=True)
        elif self.language == 'eng':
            embeddings = self.model.encode(self.catalog_eng, convert_to_tensor=True)
        else:
            raise ValueError(f"Unknown language: {self.language}")

        # compute query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # compute cosine similarity
        cosine_similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
        cosine_similarities = cosine_similarities.cpu().numpy()

        # compare the query with each item in the data source
        results = []
        for i, item_id in enumerate(self.items):
            item, score = self.items[item_id]
            similarity_score = cosine_similarities[i]
            results.append((item, similarity_score))
        
        # sort results by similarity score and limit to output_length
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:self.output_length]
        return results