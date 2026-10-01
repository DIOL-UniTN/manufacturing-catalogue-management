from qdrant_client.http.models import PointStruct, CollectionStatus, NamedVector, VectorParams, Distance, models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # Optional for progress bar
from rel_db_functions import Material
import time
import csv


qdrant_client = QdrantClient(host="localhost", port=6333)
model_old = SentenceTransformer('distiluse-base-multilingual-cased-v2')
model = SentenceTransformer('intfloat/multilingual-e5-large')

DATASETS = {
    # "catalogue_01": "../datasets/catalogue_01.csv",
    # "catalogue_02": "../datasets/catalogue_02.csv",
    # "catalogue_03": "../datasets/catalogue_03.csv"

    # "catalogue_01_e5": "../datasets/catalogue_01.csv",
    # "catalogue_02_e5": "../datasets/catalogue_02.csv",
    # "catalogue_03_e5": "../datasets/catalogue_03.csv"

    "catalogue_01_e5_mixed": "../datasets/catalogue_01.csv",
}

COLLECTION_NAME = "catalogue_01_e5_mixed"
# COLLECTION_NAME = "catalogue_01"

def create_collections():
    for collection_name in DATASETS.keys():
        try:
            if qdrant_client.collection_exists(collection_name=collection_name):
                print(f"Collection {collection_name} already created")
            else:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        # "short_desc_ita": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                        # "short_desc_eng": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                        "long_desc_ita": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                        "long_desc_eng": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    }
                )
                print(f"Collection {collection_name} created successfully")
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            return False
    return True

def create_vector_dbs():
    for collection_name, csv_path in DATASETS.items():
        print(f"\nProcessing {collection_name} from {csv_path}")
        points = []
        point_id = 1
        
        # Read CSV and count rows for progress bar
        with open(csv_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header row
        
        # Process CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in tqdm(reader, total=row_count, desc=f"Processing {collection_name}"):
                # Prepare texts
                short_it = row.get('short_it', '').strip()
                short_eng = row.get('short_eng', '').strip()
                long_it = row.get('long_it', '').strip()
                long_eng = row.get('long_eng', '').strip()
                
                # Use short description if long is empty
                long_it = ("corto: " + short_it.lower() + ", lungo:" + long_it) if long_it else ("corto: " + short_it)
                long_eng = ("short: " + short_eng.lower() + ", long:" + long_eng) if long_eng else ("short: " + short_eng)

                # long_it = long_it if long_it else short_it
                # long_eng = long_eng if long_eng else short_eng
                
                # Generate embeddings
                embeddings = {
                    # "short_desc_ita": model.encode('passage: ' + short_it.lower(), normalize_embeddings=True).tolist() if short_it else [],
                    # "short_desc_eng": model.encode('passage: ' + short_eng.lower(), normalize_embeddings=True).tolist() if short_eng else [],
                    "long_desc_ita": model.encode('passage: ' + long_it.lower(), normalize_embeddings=True).tolist() if long_it else [],
                    "long_desc_eng": model.encode('passage: ' + long_eng.lower(), normalize_embeddings=True).tolist() if long_eng else []
                }
                
                # Create payload
                payload = {
                    "id": row['id'],
                    "short_it": short_it,
                    "short_eng": short_eng,
                    "long_it": long_it,
                    "long_eng": long_eng
                }
                
                # Create point
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embeddings,
                    payload=payload
                ))
                point_id += 1
                
                # Batch upload
                if len(points) >= 100:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    points = []
            
            # Upload final batch
            if points:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
        
        # Verify collection
        try:
            info = qdrant_client.get_collection(collection_name)
            print(f"{collection_name} ready. Points: {info.points_count}")
        except Exception as e:
            print(f"Verification failed for {collection_name}: {e}")

# Run the process
if __name__ == "__main__":
    if create_collections():
        create_vector_dbs()


def vector_search(query, top_n=10):
    try:
        # Generate query embedding
        query_embedding = model.encode(query.lower()).tolist()
        # query_embedding = model_old.encode(query.lower(), normalize_embeddings=True).tolist()

    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return []

    start_time = time.time()
    
    # Search both vectors simultaneously
    try:
        # # Search Italian vectors
        # italian_short_results = qdrant_client.search(
        #     collection_name=COLLECTION_NAME,
        #     query_vector=NamedVector(name="short_desc_ita", vector=query_embedding),
        #     limit=top_n,
        #     with_payload=True
        # )
        
        # # Search English vectors
        # english_short_results = qdrant_client.search(
        #     collection_name=COLLECTION_NAME,
        #     query_vector=NamedVector(name="short_desc_eng", vector=query_embedding),
        #     limit=top_n,
        #     with_payload=True
        # )

        italian_long_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="long_desc_ita", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )

        english_long_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="long_desc_eng", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return []

    # Combine and deduplicate results
    combined_results = {}
    # for hit in italian_short_results + english_short_results + italian_long_results + english_long_results:
    for hit in italian_long_results + english_long_results:
        material_id = hit.payload["id"]
        if material_id not in combined_results or hit.score > combined_results[material_id].score:
            combined_results[material_id] = hit

    # Sort by score and select top N
    sorted_results = sorted(combined_results.values(), 
                           key=lambda x: x.score, 
                           reverse=True)[:top_n]

    print(f"Search completed in {time.time() - start_time:.2f} seconds")
    
    return [{
        "material_id": hit.payload["id"],
        "score": hit.score,
        "it": hit.payload.get("short_it", ""),
        "en": hit.payload.get("short_eng", ""),
    } for hit in sorted_results]