from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from retrievers.Item import Item
from tqdm import tqdm
import time
import csv
import torch

class Bge(Retriever):
    def __init__(self, data_source: str, output_length: int, return_dense: bool = True, return_sparse: bool = False):
        super().__init__(data_source, output_length)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        if return_dense and return_sparse:
            raise ValueError("Cannot return both dense and sparse vectors. Choose one of them.")
        
        if return_dense:
            self.model = BGEM3FlagModel("BAAI/bge-m3", devices=device, return_dense=True, return_sparse=False)
            self.mode = "dense_vecs"
        elif return_sparse:
            self.model = BGEM3FlagModel("BAAI/bge-m3", devices=device, return_dense=False, return_sparse=True)
            self.mode = "lexical_weights"
        else:
            raise ValueError("At least one of return_dense or return_sparse must be True.")
        
        # create qdrant client and database
        #self.qdrant_client = QdrantClient(":memory:")
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        if not self.qdrant_client.collection_exists(collection_name=f"vector-database-{self.catalogue_name}-bge-{self.mode}"):
            self.qdrant_client.recreate_collection(
                collection_name=f"vector-database-{self.catalogue_name}-bge-{self.mode}",
                vectors_config={
                    "desc_ita": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "desc_eng": models.VectorParams(size=1024, distance=models.Distance.COSINE),
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
                    vector_ita = self.model.encode([it], convert_to_numpy=True)[self.mode][0]
                    vector_eng = self.model.encode([eng], convert_to_numpy=True)[self.mode][0]

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
                    if len(points) >= 100:  # upload points in batches of 100
                        self.qdrant_client.upsert(
                            collection_name=f"vector-database-{self.catalogue_name}-bge-{self.mode}",
                            points=points
                        )
                        points = []
                
            # upload points to the vector database
            if points:
                self.qdrant_client.upsert(
                    collection_name=f"vector-database-{self.catalogue_name}-bge-{self.mode}",
                    points=points
                )

        info = self.qdrant_client.get_collection(f"vector-database-{self.catalogue_name}-bge-{self.mode}")
        print(f"Vector database ready. Points: {info.points_count}")

    def retrieve(self, query:str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()
        super().retrieve(query, language=language)

        # encode the query
        query_vector = self.model.encode([query.lower()], convert_to_numpy=True)[self.mode][0]

        # search in the vector database
        results = self.qdrant_client.search(
            collection_name=f"vector-database-{self.catalogue_name}-bge-{self.mode}",
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