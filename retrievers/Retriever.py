from typing import Dict, List, Set, Tuple
from retrievers.Item import Item
import os
import argparse
from langdetect import detect
import os

class Retriever:
    def __init__(self, data_source: str, output_length: int):
        self.data_source = data_source
        self.output_length = output_length
        # get catalogue name from data source
        filename = os.path.basename(self.data_source)
        self.catalogue_name = filename[:-4]  # remove .csv extension

    def retrieve(self, query:str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        self.language = language
        if not self.language:
            # detect the language of the query
            try:
                self.language = detect(query)
                if self.language not in ['it', 'en']:
                    raise ValueError(f"Unsupported language detected: {self.language}. Supported languages are 'it' and 'en'.")
                elif self.language == 'it':
                    self.language = 'ita'
                elif self.language == 'en':
                    self.language = 'eng'
            except Exception as e:
                print(f"Error detecting language: {e}")
                self.language = 'ita'
    
def read_arguments():
    parser = argparse.ArgumentParser(description="Python class that retrieves data based on a user query.")
    
    parser.add_argument('--catalogue', type=str, required=True, help='Path to the data source file.')
    parser.add_argument('--query', type=str, required=True, help='User query.')
    parser.add_argument('--model', type=str, required=True, choices=['bm25', 'sbert_512', 'sbert_768', 'sbert_1024', 'qwen_1024', 'qwen_2560', 'qwen_4096', 'random'], help='Algorithm model to use for retrieval [bm25, sbert_512, sbert_1024, random].')
    parser.add_argument('--output_length', type=int, required=True, help='Number of results to return.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    # retrievers
    from Bm25 import Bm25
    from Sbert import Sbert
    from Qwen import Qwen
    from Random import Random

    args = read_arguments()
    catalogue = args['catalogue']
    query = args['query']
    model = args['model']
    output_length = args['output_length']

    if model == 'bm25':
        retriever = Bm25(catalogue, output_length)
    elif model == 'sbert_512':
        retriever = Sbert(catalogue, output_length, 512)
    elif model == 'sbert_768':
        retriever = Sbert(catalogue, output_length, 768)
    elif model == 'sbert_1024':
        retriever = Sbert(catalogue, output_length, 1024)
    elif model == 'qwen_1024':
        retriever = Qwen(catalogue, output_length, 1024)
    elif model == 'qwen_2560':
        retriever = Qwen(catalogue, output_length, 2560)
    elif model == 'qwen_4096':
        retriever = Qwen(catalogue, output_length, 4096)
    elif model == 'random':
        retriever = Random(catalogue, output_length, random_seed=123)
    else:
        raise ValueError(f"Unknown model: {model}. Supported models are: bm25, sbert_512, sbert_1024, random.")
    
    results, time_taken = retriever.retrieve(query)
    for i,r in enumerate(results):
        print(f"[{i+1}] item: {r[0]}, score: {r[1]}")
    print(f"Time taken: {time_taken:.4f} seconds")