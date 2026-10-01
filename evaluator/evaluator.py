from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import pandas as pd
import os
import sys
import argparse
import datetime
# retriever_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../retrievers'))
# sys.path.append(retriever_path)
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
from retrievers.HybridRetriever import HybridRetriever

def get_model(model_name: str, catalogue: str, output_length: int) -> Retriever:
    if model_name == 'bm25':
        return Bm25(data_source=catalogue, output_length=output_length)
    elif model_name == 'sbert_512':
        return Sbert(data_source=catalogue, output_length=output_length, size=512)
    elif model_name == 'sbert_768':
        return Sbert(data_source=catalogue, output_length=output_length, size=768)
    elif model_name == 'sbert_1024':
        return Sbert(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'qwen_1024':
        return Qwen(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'qwen_2560':
        return Qwen(data_source=catalogue, output_length=output_length, size=2560)
    elif model_name == 'qwen_4096':
        return Qwen(data_source=catalogue, output_length=output_length, size=4096)
    elif model_name == 'bge_dense':
        return Bge(data_source=catalogue, output_length=output_length, return_dense=True, return_sparse=False)
    elif model_name == 'bge_sparse':
        return Bge(data_source=catalogue, output_length=output_length, return_dense=False, return_sparse=True)
    elif model_name == 'gte':
        return Gte(data_source=catalogue, output_length=output_length)
    elif model_name == 'nomic':
        return Nomic(data_source=catalogue, output_length=output_length)
    elif model_name == 'random':
        return Random(data_source=catalogue, output_length=output_length, random_seed=123)
    elif model_name == 'fuzzy_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='ratio')
    elif model_name == 'fuzzy_sort_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_sort_ratio')
    elif model_name == 'fuzzy_set_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_set_ratio')
    elif model_name == 'tfidf':
        return Tfidf(data_source=catalogue, output_length=output_length)
    elif model_name == 'hybrid_sbert_1024_miniL6':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='cross_encoder', model_name='ms-marco-MiniLM-L-6-v2')
    elif model_name == 'hybrid_sbert_1024_bm25':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='bm25')
    elif model_name == 'hybrid_sbert_1024_fuzzy':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='fuzzy', model_name='token_sort_ratio')
    elif model_name == 'hybrid_bm25_miniL6':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='bm25', ranker_name='cross_encoder', model_name='ms-marco-MiniLM-L-6-v2')
    elif model_name == 'hybrid_sbert_1024_miniL6':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='cross_encoder', model_name='ms-marco-MiniLM-L-6-v2')
    elif model_name == 'hybrid_sbert_1024_bge_m3':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='bge_m3')
    elif model_name == 'hybrid_sbert_1024_bge_gemma':
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', ranker_name='bge_gemma')
    elif model_name == "hybrid_sbert_1024_tfidf":
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', retriever_length=100, ranker_name='tfidf_ranker')
    elif model_name == "hybrid_tfidf_sbert_1024":
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='tfidf', retriever_length=100, ranker_name='sbert_1024_ranker')
    else:
        raise ValueError(f"Unknown model: {model_name}")

def compute_map_k(model: Retriever,
                  dataset: pd.DataFrame,
                  language: str = None,
                  output_file: str = None,
                  output_file_retriver_error: str = None,
                  output_length: int = 10) -> Tuple[float, float, float]:
    scores1 = []
    scores10 = []
    times = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Computing MAP@1 and MAP@10"):
        query = row['query']
        positive_id = row['positive']
        hard_negative_id = row['hard_negative']
        soft_negative_id = row['soft_negative']

        positive_position = -1
        hard_negative_position = -1
        soft_negative_position = -1

        results, retrive_time = model.retrieve(query=query, language=language)
        times.append(retrive_time)

        ids = [item.item_id for item, _ in results]
        if positive_id in ids:
            positive_position = ids.index(positive_id)
        if hard_negative_id in ids:
            hard_negative_position = ids.index(hard_negative_id)
        if soft_negative_id in ids:
            soft_negative_position = ids.index(soft_negative_id)

        map_ids = [item.item_id for item, _ in results[0:10]]
        if positive_id == map_ids[0]:
            scores1.append(1.0)
        else:
            scores1.append(0.0)

        if positive_id in map_ids:
            rank = map_ids.index(positive_id) + 1
            scores10.append(1.0 / rank)
        else:
            scores10.append(0.0)

        # Write positions to the output file
        if ',' in query:
            query_to_write = f'"{query}"'
        else:
            query_to_write = query
        
        with open(output_file, 'a') as f:
            f.write(f"{query_to_write},{output_length},{positive_id},{hard_negative_id},{soft_negative_id},{positive_position},{hard_negative_position},{soft_negative_position}\n")

        # Write retriever error if positive document is not in the first position
        if positive_position != 0:
            with open(output_file_retriver_error, 'a') as f:
                f.write(f"{query_to_write},{output_length},{positive_id},{positive_position}\n")

    map_1 = sum(scores1) / len(scores1) if scores1 else 0.0
    map_10 = sum(scores10) / len(scores10) if scores10 else 0.0
    avg_time = sum(times) / len(times)
    return {"score_1": map_1, "score_10": map_10, 'avg_time': avg_time}

def evaluate_model_on_datasets(dataset_paths: List[str],
                               dataset_name: str,
                               catalogue_paths: List[str],
                               model_name: str,
                               language: str = None,
                               output_file: str = None,
                               output_file_retriver_error: str = None,
                               output_length: int = 10) -> List[Tuple[str, float, float, float]]:
    results = []
    for dataset_path, catalogue_path in zip(dataset_paths, catalogue_paths):
        dataset = pd.read_csv(dataset_path, sep=',')
        dataset = dataset.head(100) # limit to 100 rows for testing
        model = get_model(model_name, catalogue_path, output_length)
        score_data = compute_map_k(model, dataset, language=language, output_file=output_file, output_file_retriver_error=output_file_retriver_error, output_length=output_length)
        results.append((dataset_name, score_data['score_1'], score_data['score_10'], score_data['avg_time']))
    return results

def read_arguments():
    parser = argparse.ArgumentParser(description="Python to evaluate a retriever with respect to a dataset and a catalogue.")
    
    parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--catalogue', type=str, required=True, help='Path of the catalogue.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--model', type=str, required=True, choices=['bm25',
                                                                     'sbert_512',
                                                                     'sbert_768',
                                                                     'sbert_1024',
                                                                     'qwen_1024',
                                                                     'qwen_2560',
                                                                     'qwen_4096',
                                                                     'bge_dense',
                                                                     'bge_sparse',
                                                                     'gte',
                                                                     'nomic',
                                                                     'random',
                                                                     'fuzzy_ratio',
                                                                     'fuzzy_sort_ratio',
                                                                     'fuzzy_set_ratio',
                                                                     'tfidf',
                                                                     'hybrid_sbert_1024_bge_m3',
                                                                     'hybrid_sbert_1024_bge_gemma',
                                                                     'hybrid_sbert_1024_tfidf',
                                                                     'hybrid_tfidf_sbert_1024'], help='Algorithm model to use for retrieval.')
    parser.add_argument('--language', type=str, default=None, help='Language of the dataset (optional,).', choices=['ita', 'eng'])
    parser.add_argument('--output_length', type=int, default=10, help='Number of results to return from the retriever (default: 10).')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    dataset_path = args['dataset']
    dataset_name = args['dataset_name']
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{args['output_folder']}/{current_datetime}"
    catalogue_path = args['catalogue']
    model = args['model']
    language = args['language']
    output_length = args['output_length']

    os.makedirs(output_folder, exist_ok=True)

    # Create csv to store positions of the positive document
    output_file = os.path.join(output_folder, f"positions_{model}.csv")
    with open(output_file, 'w') as f:
        f.write('query,output_length,positive,hard_negative,soft_negative,positive_position,hard_negative_position,soft_negative_position\n')
    
    # Create csv to store the queries for which the retriver was not able to retrieve the positive document in the first position
    output_file_retriver_error = os.path.join(output_folder, f"retriever_error_{model}.csv")
    with open(output_file_retriver_error, 'w') as f:
        f.write('query,output_length,positive,positive_position\n')

    results = evaluate_model_on_datasets(
        dataset_paths=[dataset_path],
        dataset_name=dataset_name,
        catalogue_paths=[catalogue_path],
        model_name=model,
        language=language,
        output_file=output_file,
        output_file_retriver_error=output_file_retriver_error,
        output_length=output_length
    )

    # print the results
    for dataset, map1, map10, avg_time in results:
        print(f"Dataset: {dataset}, MAP@{1}: {map1:.4f}, MAP@{10}: {map10:.4f}, Avg Time: {avg_time:.4f} seconds")

    # Save MAP@1 results: Dataset, MAP@1, Avg Time (seconds)
    output_file_map1 = os.path.join(output_folder, f"results_{model}_k_1.csv")
    results_df_map1 = pd.DataFrame(
        [(dataset, map1, avg_time) for dataset, map1, _, avg_time in results],
        columns=['Dataset', 'MAP@1', 'Avg Time (seconds)']
    )
    results_df_map1.to_csv(output_file_map1, index=False)

    # Save MAP@10 results: Dataset, MAP@10, Avg Time (seconds)
    output_file_map10 = os.path.join(output_folder, f"results_{model}_k_10.csv")
    results_df_map10 = pd.DataFrame(
        [(dataset, map10, avg_time) for dataset, _, map10, avg_time in results],
        columns=['Dataset', 'MAP@10', 'Avg Time (seconds)']
    )
    results_df_map10.to_csv(output_file_map10, index=False)
