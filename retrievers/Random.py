from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from retrievers.Item import Item
import time
import csv
import random

class Random(Retriever):
    def __init__(self, data_source: str, output_length: int, random_seed: int, language: str = None):
        super().__init__(data_source, output_length, language=language)

        # set the random seed for reproducibility
        self.rng = random.seed(random_seed)

        # populate a list with all materials from the data 
        self.items = []
        with open(self.data_source, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                item = Item(
                    item_id=row['id'],
                    ita_short_desc=row['short_ita'].strip(),
                    eng_short_desc=row['short_eng'].strip(),
                    ita_long_desc=row['long_ita'].strip() if row['long_ita'] else None,
                    eng_long_desc=row['long_eng'].strip() if row['long_eng'] else None
                )
                self.items.append(item)
        
        print(f"Random retriever initialized with {len(self.items)} items from {self.data_source}")

    def retrieve(self, query:str) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()

        # randomly select items
        selected_items = random.sample(self.items, min(self.output_length, len(self.items)))

        # create results with dummy scores
        results = [(item, random.uniform(0, 1)) for item in selected_items]

        end = time.time()
        time_taken = end - start

        return results, time_taken