from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from retrievers.Item import Item
from tqdm import tqdm
import time
import csv
import torch

class Qwen(Retriever):
    def __init__(self, data_source: str, output_length: int, size: int):
        super().__init__(data_source, output_length)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.size = size

        # init model
        if self.size == 1024:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=device)
        elif self.size == 2560:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-4B', device=device)
        elif self.size == 4096:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', device=device)
        else:
            raise ValueError(f"Unknown model size: {size}")
        
        # create qdrant client and database
        #self.qdrant_client = QdrantClient(":memory:")
        timeout = None if self.size != 4096 else 60.0  # adjust timeout based on model size
        print(f"Using timeout: {timeout} seconds for model size {self.size}")
        self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=timeout)
        if not self.qdrant_client.collection_exists(collection_name=f"vector-database-{self.catalogue_name}-qwen-{self.size}"):
            self.qdrant_client.recreate_collection(
                collection_name=f"vector-database-{self.catalogue_name}-qwen-{self.size}",
                vectors_config={
                    "desc_ita": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                    "desc_eng": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                }
            )

            # populate the vector database with items from the catalogue
            points = []
            point_id = 1
            # read CSV and count rows for progress bar
            with open(self.data_source, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1
            with open(self.data_source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in tqdm(reader, total=row_count, desc="Populating vector database"):
                    # prepare texts
                    short_ita = row.get('short_ita', '').strip()
                    short_eng = row.get('short_eng', '').strip()
                    long_ita = row.get('long_ita', '').strip()
                    long_eng = row.get('long_eng', '').strip()

                    it = "corta: " + short_ita 
                    if long_ita:
                        it += f" | lunga: {long_ita}"
                    eng = "short: " + short_eng
                    if long_eng:
                        eng += f" | long: {long_eng}"

                    # use short description if long is empty
                    if not long_ita:
                        long_ita = short_ita
                    if not long_eng:
                        long_eng = short_eng

                    # create vectors
                    vector_ita = self.model.encode(it, convert_to_numpy=True)
                    vector_eng = self.model.encode(eng, convert_to_numpy=True)

                    point = models.PointStruct(
                        id=point_id,
                        vector={
                            "desc_ita": vector_ita,
                            "desc_eng": vector_eng,
                        },
                        payload={
                            "item_id": row['id'],
                            "short_ita": short_ita,
                            "short_eng": short_eng,
                            "long_ita": long_ita,
                            "long_eng": long_eng
                        }
                    )
                    points.append(point)
                    point_id += 1
                    batch_size = 100 if self.size != 4096 else 20  # adjust batch size based on model size
                    if len(points) >= batch_size:
                        self.qdrant_client.upsert(
                            collection_name=f"vector-database-{self.catalogue_name}-qwen-{self.size}",
                            points=points
                        )
                        points = []
                
            # upload points to the vector database
            if points:
                self.qdrant_client.upsert(
                    collection_name=f"vector-database-{self.catalogue_name}-qwen-{self.size}",
                    points=points
                )

        info = self.qdrant_client.get_collection(f"vector-database-{self.catalogue_name}-qwen-{self.size}")
        print(f"Vector database ready. Points: {info.points_count}")

    def retrieve(self, query:str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()
        super().retrieve(query, language=language)

        # encode the query
        query_vector = self.model.encode(query.lower(), convert_to_numpy=True)

        # search in the vector database
        results = self.qdrant_client.search(
            collection_name=f"vector-database-{self.catalogue_name}-qwen-{self.size}",
            query_vector=models.NamedVector(name=f"desc_{self.language}", vector=query_vector),
            limit=self.output_length
        )

        items = []
        for result in results:
            item_id = result.payload["item_id"]
            score = result.score

            item = Item(item_id=item_id, 
                        ita_short_desc=result.payload["short_ita"],
                        eng_short_desc=result.payload["short_eng"],
                        ita_long_desc=result.payload["long_ita"],
                        eng_long_desc=result.payload["long_eng"])
            items.append((item, score))

        retrieve_time = time.time() - start
        return items, retrieve_time