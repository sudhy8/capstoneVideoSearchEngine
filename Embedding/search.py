# from transformers import CLIPProcessor, CLIPModel,PreTrainedTokenizerFast
from transformers import PreTrainedTokenizerFast, CLIPProcessor, CLIPModel,CLIPTokenizerFast
import torch
import weaviate.classes as wvc

device = "cuda" if torch.cuda.is_available() else \
("mps" if torch.backends.mps.is_available() else "cpu")

device = "cuda" if torch.cuda.is_available() else \
("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"

model = CLIPModel. from_pretrained(model_id) . to(device)
tokenizer = CLIPTokenizerFast. from_pretrained(model_id)
processor = CLIPProcessor. from_pretrained(model_id)

import weaviate
import os

with weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", "https://ccc-q2m6s9m3.weaviate.network"),  # Replace with your WCS URL
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "imlIdfNFZ8eC6rm4iCvSrTJyQeoBo6KjQtLT"))  # Replace with your WCS key
) as client:  # Use this context manager to ensure the connection is closed
    print(client.is_ready())


def finder():
    client.connect()

    names = client.collections.get("videoSearch")

    inputs = tokenizer("hill", return_tensors='pt')
    text_emb = model.get_text_features(**inputs)

    names = client.collections.get("videoSearch")

    response = names.query.near_vector(
            near_vector=list(text_emb[0]),
            limit=4,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )

    for obj in response.objects:
        name = obj.properties['name']

        print(f"Name: {name}")


finder()