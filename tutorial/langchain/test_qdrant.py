#%%
from dotenv import load_dotenv
load_dotenv('.env')


import pandas as pd
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams


# %%
def search_results_to_dataframe(results):
    """
    Converts QdrantVectorStore search results to a pandas DataFrame.

    Args:
    results (list): List of tuples where each tuple contains a dictionary with the key 'text' and a similarity score.

    Returns:
    pd.DataFrame: DataFrame with columns 'text' and 'score'.
    """
    lst = pd.DataFrame([{**res.metadata, 'content': res.page_content, 'score':score} for res, score in results])
    df = pd.DataFrame(lst)

    return df

def create_fack_docs():
    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    return documents, uuids

documents, uuids = create_fack_docs()

docs = [d.page_content for d in documents]
metadata = [d.metadata for d in documents]
docs

#%%

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

class VectoreDB:
    def __init__(self, cfg, collection_name, embed_dim):
        self.client = QdrantClient(**cfg)
        self.collection_name = collection_name

        self.create_collection(
            collection_name = self.collection_name,
            vectors_config = VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    def create_collection(self, collection_name, vectors_config):
        """
        Create a collection in Qdrant if it does not exist.
        """
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )

    def add(self, points):
        """
        Add or upsert points into the collection.

        :param points: A list of PointStruct to be added.
        :return: Operation info from Qdrant.
        """
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

    def delete(self, point_ids):
        """
        Delete points from the collection by their IDs.

        :param point_ids: A list of point IDs to delete.
        :return: Operation info from Qdrant.
        """
        return self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids
        )

    def modify(self, uuid, *args, **kwargs):
        """
        Modify the payload of a vector in the collection.

        :param uuid: The ID of the vector to modify.
        :param args: List of key-value pairs to update in the payload.
        :param kwargs: Dictionary of key-value pairs to update in the payload.
        :return: The updated payload.
        """
        existing_points = self.client.retrieve(collection_name=self.collection_name, ids=[uuid])

        if not existing_points:
            raise ValueError(f"No vector found with ID {uuid} in collection {self.collection_name}.")

        existing_payload = existing_points[0].payload

        for key, value in args:
            existing_payload[key] = value

        existing_payload.update(kwargs)

        self.client.set_payload(
            collection_name=self.collection_name,
            payload=existing_payload,
            points=[uuid]
        )

        return existing_payload

    def query(self, query_vector, limit=10):
        """
        Query the collection using a vector.

        :param query_vector: The vector to search for.
        :param limit: The maximum number of results to return.
        :return: A list of search results.
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        # return [result.dict() for result in search_result]
        payloads = [hit.payload for hit in search_result]

        return payloads

if __name__ == "__main__":

    client_cfg = {'location': ":memory:"} # path="./langchain_qdrant"
    collection_name = "demo_collection"

    client = VectoreDB(client_cfg, collection_name, 4)

    # insert
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ]

    client.add(points)

    client.delete([4])

    client.modify(uuid=3, city='深圳')

    res = client.query(query_vector=[0.2, 0.1, 0.9, 0.7], limit=6)
    pd.DataFrame(res)

