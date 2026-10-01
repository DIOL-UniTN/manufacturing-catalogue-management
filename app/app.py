from typing import Dict, List, Set, Tuple
from flask import Flask, render_template, request, jsonify
from vec_db_functions import vector_search
from insertion import get_suggested_descriptions
from retrievers.Retriever import Retriever
import configparser
import os

app = Flask(__name__)

def read_config()->Tuple[str, str, int, str]:
    config = configparser.ConfigParser()

    # determine the absolute path to config.ini relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.ini")
    config.read(config_path)
    print(f"Configuration loaded: {config.sections()}")

    settings = config["settings"]
    catalogue = settings["catalogue"]
    model = settings["retriever"]
    output_length = int(settings["output_length"])
    insertion_llm = settings["insertion_llm"]
    language = settings["language"]

    return catalogue, model, output_length, insertion_llm, language

def initialize_retriever(model_name: str, catalogue: str, output_length: int) -> Retriever:
    from retrievers.Bm25 import Bm25
    from retrievers.Sbert import Sbert
    from retrievers.Qwen import Qwen
    from retrievers.Random import Random
    from retrievers.Gte import Gte
    from retrievers.Nomic import Nomic
    from retrievers.BGE import Bge
    from retrievers.Fuzzy import Fuzzy
    from retrievers.HybridRetriever import HybridRetriever

    if model_name == 'bm25':
        return Bm25(data_source=catalogue, output_length=output_length)
    elif model_name == 'sbert_512':
        return Sbert(data_source=catalogue, output_length=output_length, size=512)
    elif model_name == 'sbert_768':
        return Sbert(data_source=catalogue, output_length=output_length, size=768)
    elif model_name == 'sbert_1024':
        return Sbert(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'bge':
        return Bge(data_source=catalogue, output_length=output_length, return_dense=True, return_sparse=False)
    elif model_name == 'qwen_1024':
        return Qwen(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'qwen_2560':
        return Qwen(data_source=catalogue, output_length=output_length, size=2560)
    elif model_name == 'qwen_4096':
        return Qwen(data_source=catalogue, output_length=output_length, size=4096)
    elif model_name == 'fuzzy_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='ratio')
    elif model_name == 'fuzzy_sort_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_sort_ratio')
    elif model_name == 'fuzzy_set_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_set_ratio')
    elif model_name == 'gte':
        return Gte(data_source=catalogue, output_length=output_length)
    elif model_name == 'nomic':
        return Nomic(data_source=catalogue, output_length=output_length)
    elif model_name == 'random':
        return Random(data_source=catalogue, output_length=output_length, random_seed=123)
    elif model_name == "hybrid_sbert_1024_tfidf":
        return HybridRetriever(data_source=catalogue, output_length=output_length, retriever_name='sbert_1024', retriever_length=100, ranker_name='tfidf_ranker')
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: bm25, sbert_512, sbert_768, sbert_1024, qwen_1024, qwen_2560, qwen_4096, gte, nomic, random.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_results', methods=['POST'])
def get_results():
    # get user input from the request
    data = request.get_json()
    user_input = data.get('input', '')
    
    print(f"User input received: {user_input}")

    # load configuration from config.ini
    catalogue, model_name, output_length, _, language = read_config()
    print(f"Using catalogue: {catalogue}, model: {model_name}, output length: {output_length}")
    # initialize the retriever based on the model name
    model = initialize_retriever(model_name, catalogue, output_length)
    results = model.retrieve(user_input, language=language)

    print(f"Time taken to retrieve results: {results[1]}")

    formatted_results = [
        {
            'id': item[0].item_id,
            'description_ita': item[0].ita_short_desc,
            'description_eng': item[0].eng_short_desc,
            'score': "{:.4f}".format(item[1]),
        }
        for item in results[0]
    ]

    print(formatted_results)
    return jsonify(formatted_results)


@app.route('/insertion')
def insert():
    return render_template('insertion.html')


@app.route('/submit_insertion', methods=['POST'])
def submit_insertion():
    data = request.get_json()
    code = data.get('code', '')
    desc_it = data.get('desc_it', '')
    desc_en = data.get('desc_en', '')
    
    return jsonify("received insertion data", code, desc_it, desc_en)

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    # get user input from the request
    data = request.get_json()
    user_input = data.get('input', '')

    # load configuration from config.ini
    catalogue, model_name, output_length, insertion_llm, language = read_config()

    # retrieve results using the vector search
    model = initialize_retriever(model_name, catalogue, output_length)
    retrived_items, _ = model.retrieve(query=user_input, language=language)

    # get the suggested descriptions using the LLM
    (ita, eng) = get_suggested_descriptions(user_input=user_input, materials=retrived_items, model=insertion_llm)

    return jsonify({'ita': ita, 'eng': eng})

if __name__ == '__main__':
    app.run()