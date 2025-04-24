import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.environ["WEAVIATE_REST_ENDPOINT"]
weaviate_api_key = os.environ["WEAVIATE_ADMIN_API_KEY"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())  # Should print: `True`

print(client.collections.list_all())  # Should print: `['GitBookChunk']`

try:
    chunks = client.collections.get("GitBookChunk")

    results = chunks.query.near_text(
        query="What is the history of git?",
        limit=3,
    )

except weaviate.exceptions.UnexpectedStatusCodeException as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
    print("Connection closed.")
