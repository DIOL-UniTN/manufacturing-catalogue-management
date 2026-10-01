from retrievers.Item import Item
from rankers.Ranker import Ranker
from typing import List, Tuple
from thefuzz import fuzz
from thefuzz import process

class FuzzyRanker(Ranker):
    def __init__(self, output_length: int, method: str):
        super().__init__(output_length)

        if method not in ['token_sort_ratio', 'token_set_ratio', 'ratio']:
            raise ValueError("Method must be one of 'token_sort_ratio', 'token_set_ratio', or 'ratio'.")
        self.method = method

    def rank(self, documents: List[Tuple[Item, float]], query: str, language: str) -> List[Tuple[Item, float]]:
        super().rank(documents, query, language)
        docs = []
        for item, _ in documents:
            if self.language == 'ita':
                desc = f"corta: {item.ita_short_desc}"
                if item.ita_long_desc:
                    desc += f" | lunga: {item.ita_long_desc}"
            elif self.language == 'eng':
                desc = f"short: {item.eng_short_desc}"
                if item.eng_long_desc:
                    desc += f" | long: {item.eng_long_desc}"

            docs.append((item.item_id, desc, ''))
        
        corpus = [doc[1].lower() for doc in docs]
        if self.method == 'token_sort_ratio':
            retriever = process.extract(query.lower(), corpus, scorer=fuzz.token_sort_ratio, limit=self.output_length)
        elif self.method == 'token_set_ratio':
            retriever = process.extract(query.lower(), corpus, scorer=fuzz.token_set_ratio, limit=self.output_length)
        elif self.method == 'ratio':
            retriever = process.extract(query.lower(), corpus, scorer=fuzz.ratio, limit=self.output_length)

        output = []
        seen = set()
        for item in retriever:
            r = item[0]
            score = item[1] 
            if len(output) >= self.output_length:
                break
            for d in docs:
                if (d[1].lower() == r or d[2].lower() == r) and d[0] not in seen:
                    item_obj = next(it for it, _ in documents if it.item_id == d[0])
                    output.append((item_obj, score))
                    seen.add(d[0])
                    break
        
        return output