from typing import Any, Dict, List, Tuple
import streamlit as st
import os
import google.generativeai as genai
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta
import couchbase.search as search
from couchbase.options import SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch


def generate_embeddings(input_data):
    """Google Generative AI를 사용하여 입력 데이터의 임베딩을 생성합니다"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=input_data,
        task_type="retrieval_query",
    )
    return result['embedding']


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@st.cache_resource
def create_filter(
    year_range: Tuple[int], rating: float, search_in_title: bool, title: str
) -> Dict[str, Any]:
    """Create a filter for the hybrid search"""
    # Fields in the document used for search
    year_field = "Released_Year"
    rating_field = "IMDB_Rating"
    title_field = "Series_Title"

    filter = {}
    filter_operations = []
    if year_range:
        year_query = {
            "min": year_range[0],
            "max": year_range[1],
            "inclusive_min": True,
            "inclusive_max": True,
            "field": year_field,
        }
        filter_operations.append(year_query)
    if rating:
        filter_operations.append(
            {
                "min": rating,
                "inclusive_min": False,
                "field": rating_field,
            }
        )
    if search_in_title:
        filter_operations.append(
            {
                "match_phrase": title,
                "field": title_field,
            }
        )
    filter["query"] = {"conjuncts": filter_operations}
    return filter


def search_couchbase(
    db_scope: Any,
    index_name: str,
    embedding_key: str,
    search_text: str,
    k: int = 5,
    fields: List[str] = ["*"],
    search_options: Dict[str, Any] = {},
):
    """Hybrid search using Python SDK in couchbase"""
    # Generate vector embeddings to search with
    search_embedding = generate_embeddings(search_text)

    # Create the search request
    search_req = search.SearchRequest.create(
        VectorSearch.from_vector_query(
            VectorQuery(
                embedding_key,
                search_embedding,
                k,
            )
        )
    )

    docs_with_score = []

    try:
        # Perform the search
        search_iter = db_scope.search(
            index_name,
            search_req,
            SearchOptions(
                limit=k,
                fields=fields,
                raw=search_options,
            ),
        )

        # Parse the results
        for row in search_iter.rows():
            score = row.score
            docs_with_score.append((row.fields, score))
    except Exception as e:
        raise e

    return docs_with_score


if __name__ == "__main__":
    st.set_page_config(
        page_title="Movie Search",
        page_icon="🎥",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Load environment variables
    DB_CONN_STR = os.getenv("DB_CONN_STR")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_BUCKET = os.getenv("DB_BUCKET")
    DB_SCOPE = os.getenv("DB_SCOPE")
    DB_COLLECTION = os.getenv("DB_COLLECTION")
    INDEX_NAME = os.getenv("INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # Initialize empty filters
    search_filters = {}

    # Google Generative AI 설정
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Connect to Couchbase Vector Store
    cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)
    bucket = cluster.bucket(DB_BUCKET)
    scope = bucket.scope(DB_SCOPE)

    # UI Elements
    text = st.text_input("Find your movie")
    with st.sidebar:
        st.header("Search Options")
        no_of_results = st.number_input(
            "Number of results", min_value=1, value=5, format="%i"
        )
        # Filters
        st.subheader("Filters")

        year_range = st.slider("Released Year", 1900, 2024, (1900, 2024))
        rating = st.number_input("Minimum IMDB Rating", 0.0, 10.0, 0.0, step=1.0)
        search_in_title = st.checkbox("Search in Title?")
        show_filter = st.checkbox("Show filter")
        hybrid_search_filter = create_filter(
            year_range, rating, search_in_title, text
        )
        if show_filter:
            st.json(hybrid_search_filter)

    if text:
        with st.spinner("Searching..."):
            # Translate the search text to English if it's in Korean
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Translate the following text to English. Only return the translated text without any additional comments: {text}",
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=100,
                    temperature=0.2,
                ),
            )
            search_text = response.text.strip()
            st.write(f"Translated text: {search_text}")

            # Fetch the filters
            search_filters = create_filter(year_range, rating, search_in_title, text)

            # Search using the Couchbase Python SDK
            results = search_couchbase(
                scope,
                INDEX_NAME,
                "Overview_embedding",
                search_text,
                k=no_of_results,
                search_options=search_filters,
            )

            for doc in results:
                movie, score = doc

                # Display the results in a grid
                st.header(movie["Series_Title"])
                col1, col2 = st.columns(2)
                with col1:
                    st.image(movie["Poster_Link"], use_column_width=True)
                with col2:
                    st.write(movie["Overview"])

                    # Translate the movies text to Korean
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(
                            f"Translate the following text to Korean. Only return the translated text without any additional comments: {movie['Overview']}",
                            generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=100,
                                temperature=0.2,
                            ),
                        )
                        st.write(response.text.strip())
                    except Exception as e:
                        print(f"Error translating text: {e}")

                    st.write(f"Score: {score:.{3}f}")
                    st.write("Released Year:", movie["Released_Year"])
                    st.write("IMDB Rating:", movie["IMDB_Rating"])
                    st.write("Runtime:", movie["Runtime"])
                st.divider()
