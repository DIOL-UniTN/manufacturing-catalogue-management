from typing import List, Tuple, Dict
from retrievers.Item import Item
from rankers.Ranker import Ranker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class TfidfRanker(Ranker):
    def __init__(self, output_length: int):
        super().__init__(output_length)

    def normalize_text(self, text: str) -> str:
        # normalize text by removing special characters and converting to lowercase
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def rank(self, documents: List[Tuple[Item, float]], query: str, language: str) -> List[Tuple[Item, float]]:
        super().rank(documents, query, language)

        # read the documents and populate python dictionary of items
        # populate catalog_ita or catalog_eng with normalized text
        self.items: Dict[str, Tuple[Item, float]] = {}
        self.catalog_ita: List[str] = []
        self.catalog_eng: List[str] = []
        for item, score in documents:
            self.items[item.item_id] = (item, score)

            if self.language == 'ita':
                text = item.ita_short_desc + ' ' + (item.ita_long_desc or '')
                self.catalog_ita.append(self.normalize_text(text))
            elif self.language == 'eng':
                text = item.eng_short_desc + ' ' + (item.eng_long_desc or '')
                self.catalog_eng.append(self.normalize_text(text))
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        
        # create TF-IDF matrix for both Italian and English catalogs
        tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
        if self.language == 'ita':
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.catalog_ita)
        elif self.language == 'eng':
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.catalog_eng)

        # normalize the query
        query_normalized = self.normalize_text(query)
        query_vector = tfidf_vectorizer.transform([query_normalized])

        # compute cosine similarity
        tfidf_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        # compare the query with each item in the data source
        results = []
        for i, item_id in enumerate(self.items):
            item, score = self.items[item_id]
            similarity_score = tfidf_similarities[i]
            results.append((item, similarity_score))
        
        # sort results by similarity score and limit to output_length
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:self.output_length]

        return results