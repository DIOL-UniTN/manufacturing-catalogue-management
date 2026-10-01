from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from retrievers.Item import Item
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv
import re

class Tfidf(Retriever):
    def __init__(self, data_source: str, output_length: int):
        super().__init__(data_source, output_length)
        
        # read the data source and populate python dictionary of items
        # populate catalog_ita and catalog_eng with normalized text
        self.data_source = data_source
        self.items: Dict[str, Item] = {}
        self.catalog_ita = []
        self.catalog_eng = []
        with open(self.data_source, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader) # skip header
            for row in reader:
                item_id = row[0]
                short_desc_ita = row[1]
                short_desc_eng = row[2]
                long_desc_ita = row[3]
                long_desc_eng = row[4]
                
                item = Item(
                    item_id=item_id,
                    ita_short_desc=short_desc_ita,
                    eng_short_desc=short_desc_eng,
                    ita_long_desc=long_desc_ita,
                    eng_long_desc=long_desc_eng
                )
                self.items[item_id] = item

                text_ita = self.normalize_text(short_desc_ita + ' ' + (long_desc_ita or ''))
                text_eng = self.normalize_text(short_desc_eng + ' ' + (long_desc_eng or ''))
                self.catalog_ita.append(text_ita)
                self.catalog_eng.append(text_eng)
        
        # create TF-IDF matrix for both Italian and English catalogs
        self.vectorizer_ita = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
        self.vectorizer_eng = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
        self.tfidf_ita = self.vectorizer_ita.fit_transform(self.catalog_ita)
        self.tfidf_eng = self.vectorizer_eng.fit_transform(self.catalog_eng)

    def normalize_text(self, text: str) -> str:
        # normalize text by removing special characters and converting to lowercase
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def retrieve(self, query:str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()
        super().retrieve(query, language=language)

        # normalize the query
        query_normalized = self.normalize_text(query)

        # get the appropriate vectorizer and TF-IDF matrix
        if self.language == 'ita':
            query_vector = self.vectorizer_ita.transform([query_normalized])
            tfidf_matrix = self.tfidf_ita
        elif self.language == 'eng':
            query_vector = self.vectorizer_eng.transform([query_normalized])
            tfidf_matrix = self.tfidf_eng
        else:
            raise ValueError(f"Unknown language: {self.language}")

        # compute cosine similarity
        tfidf_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        # compare the query with each item in the data source
        results = []
        for idx, item in enumerate(self.items.values()):
            # get TF-IDF similarity from precomputed list
            tfidf_similarity = tfidf_similarities[idx]
            results.append((item, tfidf_similarity))
        
        # sort results by similarity ratio in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        # limit results to output_length
        results = results[:self.output_length]

        end = time.time()

        return results, end-start