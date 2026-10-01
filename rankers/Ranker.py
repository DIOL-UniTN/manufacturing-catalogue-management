from typing import List, Tuple
from retrievers.Item import Item

class Ranker: 
    def __init__(self, output_length: int):
        self.output_length = output_length

    def rank(self, documents: List[Tuple[Item, float]], query: str, language: str) -> List[Tuple[Item, float]]:
        if language not in ['ita', 'eng']:
            raise ValueError("Language must be either 'ita' or 'eng'.")
        self.language = language