from typing import List, Tuple
from rankers.Ranker import Ranker
from retrievers.Item import Item
from FlagEmbedding import FlagReranker, FlagLLMReranker

class BGEReranker(Ranker):
    def __init__(self, output_length: int, model_name: str) -> None:
        super().__init__(output_length)
        if model_name == "bge_m3":
            self.model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        elif model_name == "bge_gemma":
            self.model = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def rank(self, documents: List[Tuple[Item, float]], query: str, language: str) -> List[Tuple[Item, float]]:
        super().rank(documents, query, language)
        results = []
        for item, _ in documents:
            if self.language == 'ita':
                desc = f"corta: {item.ita_short_desc}"
                if item.ita_long_desc:
                    desc += f" | lunga: {item.ita_long_desc}"
            elif self.language == 'eng':
                desc = f"short: {item.eng_short_desc}"
                if item.eng_long_desc:
                    desc += f" | long: {item.eng_long_desc}"

            score = self.model.compute_score([(query, desc)])

            results.append((item, score))


        # Sort results by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results