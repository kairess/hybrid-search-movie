from dotenv import load_dotenv
import os
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta
from tqdm import tqdm
import uuid
import pandas as pd
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load environment variables
DB_CONN_STR = os.getenv("DB_CONN_STR")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_BUCKET = os.getenv("DB_BUCKET")
DB_SCOPE = os.getenv("DB_SCOPE")
DB_COLLECTION = os.getenv("DB_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MOVIES_DATASET = "imdb_top_1000.csv"

def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    print("Connecting to couchbase...")
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


def generate_embeddings(title, input_data):
    """Google Generative AI를 사용하여 입력 데이터의 임베딩을 생성합니다"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=input_data,
        task_type="retrieval_document",
        title=title,
    )
    return result['embedding']

def cleanup_poster_url(poster_url):
    """Convert from https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_UX67_CR0,0,67,98_AL_.jpg to https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg"""

    prefix = poster_url.split("_V1_")[0]
    suffix = poster_url.split("_AL_")[1]

    return prefix + suffix


try:
    cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)
    bucket = cluster.bucket(DB_BUCKET)
    scope = bucket.scope(DB_SCOPE)
    collection = scope.collection(DB_COLLECTION)
    data = pd.read_csv(MOVIES_DATASET)

    # Convert columns to numeric types
    data["Gross"] = data["Gross"].str.replace(",", "").astype(float)

    # Fill empty values
    data["Gross"] = data["Gross"].fillna(0)
    data["Certificate"] = data["Certificate"].fillna("NA")
    data["Meta_score"] = data["Meta_score"].fillna(-1)
    data["Poster_Link"] = data["Poster_Link"].apply(cleanup_poster_url)

    data_in_dict = data.to_dict(orient="records")
    print("Ingesting Data...")
    for row in tqdm(data_in_dict):
        row["Overview_embedding"] = generate_embeddings(row["Series_Title"], row["Overview"])
        doc_id = uuid.uuid4().hex
        collection.upsert(doc_id, row)

except Exception as e:
    print("Error while ingesting data", e)
