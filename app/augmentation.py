import pandas as pd
import math
from typing import List, Dict, Optional
from APIs import get_llm_explanation

# Assume get_llm_explanation(message: str) -> str is already implemented and available.

def create_prompt_it(batch: List[Dict]) -> str:
    """
    Builds a prompt for a batch of Italian entries, using both Italian and English fields
    to generate a clear, complete Italian description.
    """
    prompt = (
        "Sei un tecnico esperto in componenti industriali. "
        "Data la seguente descrizione tecnica in italiano e la traduzione in inglese, "
        "riscrivi ciascuna descrizione in una frase chiara e completa in italiano, "
        "includendo tutte le specifiche identificabili. "
        "Non inventare informazioni.\n\n"
    )
    for idx, entry in enumerate(batch, start=1):
        prompt += f"Entry {idx}\n"
        prompt += f"[ID]: {entry['id']}\n"
        prompt += f"[SHORT_IT]: {entry['short_it']}\n"
        prompt += f"[SHORT_ENG]: {entry['short_eng']}\n"
        if entry.get('long_it') and not (isinstance(entry['long_it'], float) and math.isnan(entry['long_it'])):
            prompt += f"[LONG_IT]: {entry['long_it']}\n"
        if entry.get('long_eng') and not (isinstance(entry['long_eng'], float) and math.isnan(entry['long_eng'])):
            prompt += f"[LONG_ENG]: {entry['long_eng']}\n"
        prompt += "\n"
    prompt += "Per ogni entry, produci:\n[AUGMENTED_IT]: <frase chiara e completa in italiano>\n"
    return prompt

def create_prompt_eng(batch: List[Dict]) -> str:
    """
    Builds a prompt for a batch of English entries, using both Italian and English fields
    to generate a clear, complete English description.
    """
    prompt = (
        "You are a technical writer specialized in industrial components. "
        "Given the following technical description in English and its Italian translation, "
        "rewrite each description as a clear, complete English sentence that includes all identifiable specifications. "
        "Do not invent information.\n\n"
    )
    for idx, entry in enumerate(batch, start=1):
        prompt += f"Entry {idx}\n"
        prompt += f"[ID]: {entry['id']}\n"
        prompt += f"[SHORT_EN]: {entry['short_eng']}\n"
        prompt += f"[SHORT_IT]: {entry['short_it']}\n"
        if entry.get('long_eng') and not (isinstance(entry['long_eng'], float) and math.isnan(entry['long_eng'])):
            prompt += f"[LONG_EN]: {entry['long_eng']}\n"
        if entry.get('long_it') and not (isinstance(entry['long_it'], float) and math.isnan(entry['long_it'])):
            prompt += f"[LONG_IT]: {entry['long_it']}\n"
        prompt += "\n"
    prompt += "For each entry, output:\n[AUGMENTED_EN]: <clear and complete English sentence>\n"
    return prompt

def parse_llm_response_it(response: str, batch_size: int) -> List[str]:
    """
    Parses the LLM response for Italian augmentation, extracting lines starting with [AUGMENTED_IT]:.
    Returns a list of length batch_size with the generated sentences.
    """
    lines = response.splitlines()
    augmented = []
    for line in lines:
        if line.strip().startswith("[AUGMENTED_IT]:"):
            augmented.append(line.strip().split(":", 1)[1].strip())
    # Pad if fewer items returned
    while len(augmented) < batch_size:
        augmented.append("")
    return augmented[:batch_size]

def parse_llm_response_eng(response: str, batch_size: int) -> List[str]:
    """
    Parses the LLM response for English augmentation, extracting lines starting with [AUGMENTED_EN]:.
    Returns a list of length batch_size with the generated sentences.
    """
    lines = response.splitlines()
    augmented = []
    for line in lines:
        if line.strip().startswith("[AUGMENTED_EN]:"):
            augmented.append(line.strip().split(":", 1)[1].strip())
    # Pad if fewer items returned
    while len(augmented) < batch_size:
        augmented.append("")
    return augmented[:batch_size]

def augment_batch_it(batch: List[Dict]) -> List[str]:
    """
    Given a batch of entries (as dicts), creates a prompt, calls the LLM, and returns
    a list of augmented Italian descriptions.
    """
    prompt = create_prompt_it(batch)
    response = get_llm_explanation(prompt)
    return parse_llm_response_it(response, len(batch))

def augment_batch_eng(batch: List[Dict]) -> List[str]:
    """
    Given a batch of entries (as dicts), creates a prompt, calls the LLM, and returns
    a list of augmented English descriptions.
    """
    prompt = create_prompt_eng(batch)
    response = get_llm_explanation(prompt)
    return parse_llm_response_eng(response, len(batch))

def process_csv(
    input_csv_path: str,
    output_csv_path: str,
    batch_size: int = 10,
    start_idx_it: int = 0,
    start_idx_eng: Optional[int] = None
) -> None:
    """
    Reads the input CSV (semicolon-separated) with columns:
      id;short_it;short_eng;long_it;long_eng
    Processes Italian and English augmentations in batches, with optional resume points.
    Writes out a new CSV with:
      id;short_it;short_eng;long_it;long_eng;augmented_it;augmented_eng

    Parameters:
    - start_idx_it: the row index (0-based) from which to start Italian augmentation.
      Defaults to 0. 
    - start_idx_eng: the row index (0-based) from which to start English augmentation.
      If None, it defaults to start_idx_it (so English resumes at same point). 
    """
    df = pd.read_csv(input_csv_path, sep=';', dtype=str).fillna("")
    if start_idx_eng is None:
        start_idx_eng = start_idx_it

    # Ensure augmented columns exist
    if "augmented_it" not in df.columns:
        df["augmented_it"] = ""
    if "augmented_eng" not in df.columns:
        df["augmented_eng"] = ""

    records = df.to_dict(orient="records")
    n = len(records)

    # 1. Augment Italian descriptions starting from start_idx_it
    for start in range(start_idx_it, n, batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            break
        augmented_texts = augment_batch_it(batch)
        for i, aug in enumerate(augmented_texts):
            df.at[start + i, "augmented_it"] = aug

    # 2. Augment English descriptions starting from start_idx_eng
    for start in range(start_idx_eng, n, batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            break
        augmented_texts = augment_batch_eng(batch)
        for i, aug in enumerate(augmented_texts):
            df.at[start + i, "augmented_eng"] = aug

    df.to_csv(output_csv_path, sep=';', index=False)

def process_single_batch_it(
    input_csv_path: str,
    start_idx: int,
    batch_size: int = 10
) -> List[str]:
    """
    Loads the CSV and returns the list of augmented Italian descriptions
    for the batch starting at start_idx. Does NOT write to disk.
    """
    df = pd.read_csv(input_csv_path, sep=';', dtype=str).fillna("")
    records = df.to_dict(orient="records")
    batch = records[start_idx : start_idx + batch_size]
    if not batch:
        return []
    return augment_batch_it(batch)

def process_single_batch_eng(
    input_csv_path: str,
    start_idx: int,
    batch_size: int = 10
) -> List[str]:
    """
    Loads the CSV and returns the list of augmented English descriptions
    for the batch starting at start_idx. Does NOT write to disk.
    """
    df = pd.read_csv(input_csv_path, sep=';', dtype=str).fillna("")
    records = df.to_dict(orient="records")
    batch = records[start_idx : start_idx + batch_size]
    if not batch:
        return []
    return augment_batch_eng(batch)

# Example usage:

if __name__ == "__main__":
    INPUT_CSV = "../datasets/catalogue_01.csv"
    OUTPUT_CSV = "../datasets/augmented_catalogue_01.csv"
    BATCH_SIZE = 10

    # 1. To process the entire file from scratch:
    # process_csv(INPUT_CSV, OUTPUT_CSV, batch_size=BATCH_SIZE)

    # 2. To resume Italian augmentation starting at row 50 (0-based):
    # process_csv(INPUT_CSV, OUTPUT_CSV, batch_size=BATCH_SIZE, start_idx_it=50)

    # 3. To run and inspect just one Italian batch:
    augmented_it_batch = process_single_batch_it(INPUT_CSV, start_idx=2170, batch_size=BATCH_SIZE)
    for idx, text in enumerate(augmented_it_batch, start=2170):
        print(f"Row {idx} [AUGMENTED_IT]: {text}")

    # 4. To run and inspect just one English batch:
    # augmented_eng_batch = process_single_batch_eng(INPUT_CSV, start_idx=30, batch_size=BATCH_SIZE)
    # for idx, text in enumerate(augmented_eng_batch, start=30):
    #     print(f"Row {idx} [AUGMENTED_EN]: {text}")
    pass
