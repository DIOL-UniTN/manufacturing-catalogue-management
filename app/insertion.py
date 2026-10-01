from typing import Dict, List, Set, Tuple
import time
import tiktoken
import os
import sys
import argparse
import time
import pandas as pd

# llm imports
from dotenv import load_dotenv
from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import RetryPolicy
from ollama import chat

# retriever import
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
from retrievers.Item import Item

# Load environment variables once
def _load_config():
    load_dotenv()
    return {
        "token": os.environ.get('GITHUB_TOKEN'),
        "llama_model": os.environ.get('OLLAMA_MODEL', 'llama3'),
        "endpoint": "https://models.inference.ai.azure.com",
        "temperature": 1.0,
        "max_tokens": 4000,
        "top_p": 1.0
    }

def log_print(message, log_file="log.txt", debug=None):
    if debug:
        with open(log_file, "a", encoding="utf-8") as logf:
            logf.write(message + "\n")

def get_retriever(model_name: str, catalogue: str, output_length: int) -> Retriever:
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
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_suggested_descriptions(user_input: str, materials: List[Tuple[Item, float]], model: str) -> Tuple[List[str], List[str]]:
    """Generate Italian and English descriptions based on user input and provided materials."""
    message = "You will be shown 10 examples of Italian item descriptions and their corresponding English translations. Each description is 40 characters or fewer and represents an item from a company's catalogue.\n"
    message += "Your task:\n"
    message += "Given a user's input (which may be longer than 40 characters and may not follow the same format) which can be in Italian or in English, generate:\n"
    message += "- 5 Italian descriptions and\n"
    message += "- 5 English descriptions\n"
    message += "Each must be:\n"
    message += "- No longer than 40 characters\n"
    message += "- Matched in format and style to the examples provided.\n"
    message += "- Derived from the user's input\n"
    message += "Output format:\n"
    message += "- Exactly 10 lines total\n"
    message += "- First 5 lines: Italian descriptions\n"
    message += "- Next 5 lines: English translations\n"
    message += "- One description per line\n"
    message += "- No additional text, headers, or explanation\n"
    message += "Note: These descriptions that you must generate are proposals for a new item to be added to the catalogue.\n"
    message += "Examples:\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x16-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x20-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M4x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M4x16-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x16-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x30-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x30-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x10-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x10-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x25-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x25-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x60-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x60-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M4x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M4x20-70'}\n"
    message += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x20-70'}\n"
    message += "Example user input:\n"
    message += "vite sicurezza TCBEI A2 M6x16 80\n"
    message += "Expected output:\n"
    message += "VITE SICUREZZA TCBEI+PIN A2 M6x16-80\n"
    message += "VITE SICUREZZA TCBEI+PIN A2 M6x20-80\n"  
    message += "VITE SICUREZZA TCBEI+PIN A2 M4x16-80\n"  
    message += "VITE SICUREZZA TCBEI+PIN A2 M5x16-80\n"  
    message += "VITE SICUREZZA TCBEI+PIN A2 M6x30-80\n"  
    message += "SAFETY SCREW HSBH+PIN A2 M6x16-80\n"  
    message += "SAFETY SCREW HSBH+PIN A2 M6x20-80\n"  
    message += "SAFETY SCREW HSBH+PIN A2 M4x16-80\n"  
    message += "SAFETY SCREW HSBH+PIN A2 M5x16-80\n"  
    message += "SAFETY SCREW HSBH+PIN A2 M6x30-80\n"
    message += "Provided Examples:\n"
    formatted_materials = [
        {
            'description_ita': m.ita_short_desc,
            'description_eng': m.eng_short_desc,
        }
        for m, _ in materials
    ]
    message += '\n'.join([str(item) for item in formatted_materials])
    message += "\nUser input:\n"
    message += user_input+"\n"
    message += "Your Output:\n"
    
    # load configuration parameters
    params = _load_config()
    
    # call the LLM API
    response:str = None
    if model in ("gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"):  # OpenAI-compatible
        # invoke OpenAI API
        client = OpenAI(base_url=params["endpoint"], api_key=params["token"])
        while response is None:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": message}
                ]
            ).choices[0].message.content
    elif model in ("Meta-Llama-3.1-405B-Instruct", "Mistral-Large-2411", "DeepSeek-V3-0324", "grok-3"):
        # invoke Azure API        
        retry = RetryPolicy(retry_total=3, timeout=10)
        client = ChatCompletionsClient(
            endpoint=params["endpoint"],
            credential=AzureKeyCredential(params["token"]),
            retry_policy=retry
        )
        while response is None:
            response = client.complete(
                model=model,
                messages=[UserMessage(content=message)]
            ).choices[0].message.content
    elif model == "ollama":
        # invoke Ollama
        response = chat(
            model=params["llama_model"],
            messages=[{"role": "user", "content": message}],
        ).get("message", {}).get("content", "")
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # filter out blank lines
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # ensure we have exactly 10 lines
    if len(lines) != 10:
        print(f"Warning: Expected 10 lines, got {len(lines)}")
        print(response)
    
    # return the output, handling cases where we might have fewer than 10 lines
    ita = lines[:5] if len(lines) >= 5 else lines
    eng = lines[5:10] if len(lines) >= 10 else lines[5:] if len(lines) > 5 else []
    
    return (ita, eng)

def read_arguments():
    parser = argparse.ArgumentParser(description="Python to test the insertion part.")
    
    parser.add_argument('--debug', type=bool, default=None, help='Debug mode. If True, it will print debug information also in file log.txt.')
    parser.add_argument('--query', type=str, required=True, help='User item description.')
    parser.add_argument('--catalogue', type=str, required=True, help='Path of the catalogue.')
    parser.add_argument('--llm', type=str, required=True, choices=["all",
                                                                   "gpt-4o",
                                                                   "gpt-4o-mini",
                                                                   "gpt-4.1",
                                                                   "gpt-4.1-mini",
                                                                   "Meta-Llama-3.1-405B-Instruct",
                                                                   "Mistral-Large-2411",
                                                                   "DeepSeek-V3-0324",
                                                                   "grok-3",
                                                                   "ollama"], help='LLM model to use for generating descriptions. [all] means all models will be used.')
    parser.add_argument('--retriever', type=str, required=True, choices=['bm25',
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
                                                                        'fuzzy_set_ratio'], help='Algorithm model to use for retrieval.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == "__main__":
    args = read_arguments()
    query = args['query']
    catalogue_path = args['catalogue']
    llm = args['llm']
    retriever_name = args['retriever']

    # Initialize log file
    if args["debug"]:
        with open("log.txt", "w", encoding="utf-8") as logf:
            logf.write("Log started\n")

    # retrieve the most relevant items from the catalogue
    retriever = get_retriever(retriever_name, catalogue_path, output_length=10)
    results, retrieve_time = retriever.retrieve(query=query)
    print(f"Retrieved {len(results)} items in {retrieve_time:.2f} seconds.")
    print(f"Top 10 results for query '{query}':")
    log_print(f"Retrieved {len(results)} items in {retrieve_time:.2f} seconds.", debug=args["debug"])
    log_print(f"Top 10 results for query '{query}':", debug=args["debug"])
    retrieved_summary = []
    for item, score in results:
        summary = f"Item ID: {item.item_id}, Score: {score:.4f}, ita_short_desc: {item.ita_short_desc}, eng_short_desc: {item.eng_short_desc}, ita_long_desc: {item.ita_long_desc}, eng_long_desc: {item.eng_long_desc}"
        summary_clean = summary.replace("\n", " ").replace("\r", " ")
        retrieved_summary.append(summary_clean)
        print(summary)
        print("" + "-"*50)
        log_print(summary, debug=args["debug"])
        log_print("-" * 50, debug=args["debug"])
    
    # prepare row data
    csv_row = {
        "user_input": query,
        "retriever_model": retriever_name,
        "retrieved_items": "$".join(retrieved_summary)
    }

    # get the descriptions from the LLM
    if llm == "all":
        models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "Meta-Llama-3.1-405B-Instruct",
            "Mistral-Large-2411",
            #"DeepSeek-V3-0324",
            "grok-3",
            "ollama"
        ]
        for language_model in models:
            print(f"\n--- Using model: {language_model} ---")
            log_print(f"\n--- Using model: {language_model} ---", debug=args["debug"])
            try:
                start = time.time()
                ita, eng = get_suggested_descriptions(user_input=query, materials=results, model=language_model)
                end = time.time()
                print("Italian descriptions:")
                log_print("Italian descriptions:", debug=args["debug"])
                for ita_desc in ita:
                    print(f"  {ita_desc}")
                    log_print(f"  {ita_desc}", debug=args["debug"])
                print("English descriptions:")
                log_print("English descriptions:", debug=args["debug"])
                for eng_desc in eng:
                    print(f"  {eng_desc}")
                    log_print(f"  {eng_desc}", debug=args["debug"])
                print(f"Time taken to generate descriptions: {end - start:.2f} seconds")
                log_print(f"Time taken to generate descriptions: {end - start:.2f} seconds", debug=args["debug"])

                if args["debug"]:
                    csv_row[f"{language_model}_ita"] = "$".join(ita)
                    csv_row[f"{language_model}_eng"] = "$".join(eng)

            except Exception as e:
                print(f"[ERROR] Model {language_model} failed: {e}")
                log_print(f"[ERROR] Model {language_model} failed: {e}", debug=args["debug"])
                if args["debug"]:
                    csv_row[f"{language_model}_ita"] = "[ERROR]"
                    csv_row[f"{language_model}_eng"] = "[ERROR]"
                continue
    else:
        print(f"\n--- Using model: {llm} ---")
        log_print(f"\n--- Using model: {llm} ---", debug=args["debug"])
        try:
            start = time.time()
            ita, eng = get_suggested_descriptions(user_input=query, materials=results, model=llm)
            end = time.time()
            print("Italian descriptions:")
            log_print("Italian descriptions:", debug=args["debug"])
            for ita_desc in ita:
                print(f"  {ita_desc}")
                log_print(f"  {ita_desc}", debug=args["debug"])
            print("English descriptions:")
            log_print("English descriptions:", debug=args["debug"])
            for eng_desc in eng:
                print(f"  {eng_desc}")
                log_print(f"  {eng_desc}", debug=args["debug"])
            print(f"Time taken to generate descriptions: {end - start:.2f} seconds")
            log_print(f"Time taken to generate descriptions: {end - start:.2f} seconds", debug=args["debug"])

            if args["debug"]:
                csv_row[f"{llm}_ita"] = "$".join(ita)
                csv_row[f"{llm}_eng"] = "$".join(eng)
        except Exception as e:
            print(f"[ERROR] Model {llm} failed: {e}")
            log_print(f"[ERROR] Model {llm} failed: {e}", debug=args["debug"])
            if args["debug"]:
                csv_row[f"{llm}_ita"] = "[ERROR]"
                csv_row[f"{llm}_eng"] = "[ERROR]"
    
    # save to CSV using pandas
    if args["debug"]:
        csv_file = "log.csv"
        df = pd.DataFrame([csv_row])
        df.to_csv(csv_file, index=False, mode='w')
