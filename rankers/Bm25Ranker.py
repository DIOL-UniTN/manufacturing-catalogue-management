import bm25s
from typing import List, Tuple
from retrievers.Item import Item
from rankers.Ranker import Ranker


class Bm25Ranker(Ranker):
    def __init__(self, output_length: int):
        super().__init__(output_length)
        # Initialize any necessary parameters or configurations for BM25 ranking
        
        

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
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize(corpus), show_progress=False)
        results, scores = retriever.retrieve(bm25s.tokenize(query.lower()), k=self.output_length, show_progress=False)

        # filter the results to return only unique materials
        # and limit the number of results to output_length
        output = []
        seen = set()
        for i, r in enumerate(results[0]):
            for d in docs:
                if len(output) >= self.output_length:
                    break
                if d[1].lower() == r  and d[0] not in seen:
                    item = next(it for it, _ in documents if it.item_id == d[0])
                    output.append((item, scores[0][i]))
                    seen.add(d[0])
                    break

        
        return output
