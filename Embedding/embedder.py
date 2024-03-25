# # from transformers import CLIPProcessor, CLIPModel,PreTrainedTokenizerFast
# from transformers import PreTrainedTokenizerFast, CLIPProcessor, CLIPModel,CLIPTokenizerFast
# import torch
# from PIL import Image
# from tqdm.auto import tqdm
# import numpy as np
# import weaviate
# import os
# import json


# device = "cuda" if torch.cuda.is_available() else \
# ("mps" if torch.backends.mps.is_available() else "cpu")

# device = "cuda" if torch.cuda.is_available() else \
# ("mps" if torch.backends.mps.is_available() else "cpu")
# model_id = "openai/clip-vit-base-patch32"

# model = CLIPModel. from_pretrained(model_id) . to(device)
# tokenizer = CLIPTokenizerFast. from_pretrained(model_id)
# processor = CLIPProcessor. from_pretrained(model_id)


# with weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", "https://ccc-q2m6s9m3.weaviate.network"),  # Replace with your WCS URL
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "imlIdfNFZ8eC6rm4iCvSrTJyQeoBo6KjQtLT"))  # Replace with your WCS key
# ) as client:  # Use this context manager to ensure the connection is closed
#     print(client.is_ready())

# client.connect()
# images =[Image.open('./deer.jpg'),Image.open('./hill.jpg'),Image.open('./girl.jpg'),Image.open('./elder.jpg'),Image.open('./elder2.jpg')]
# names =['deer.jpg','hill.jpg','girl.jpg','elder.jpg','elder2.jpg']


# def img_embedder(images,names):

#     batch_size = 16
#     image_arr = None

#     for i in tqdm(range(0, len(images), batch_size)):
#         # select batch of images
#         batch = images[i:i+batch_size]
#         # process and resize
#         batch = processor(
#             text=None,
#             images=batch,
#             return_tensors='pt',
#             padding=True
#         )['pixel_values'].to(device)
#         # get image embeddings
#         batch_emb = model.get_image_features(pixel_values=batch)
#         # convert to numpy array
#         batch_emb = batch_emb.squeeze(0)
#         batch_emb = batch_emb.cpu().detach().numpy()
#         # add to larger array of all image embeddings
#         if image_arr is None:
#             image_arr = batch_emb
#         else:
#             image_arr = np.concatenate((image_arr, batch_emb), axis=0)
#     image_arr.shape


#     videoSearch = client.collections.get("VideoSearch")
#     for i in zip(image_arr,names):
#         uuid =videoSearch.data.insert(
#             {
#             "name": i[1]
#         },
#         vector = i[0].tolist()
#         )
#         print(uuid)



# img_embedder(images,names)


from transformers import PreTrainedTokenizerFast, CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import weaviate
import os
import json

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

with weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", "https://ccc-q2m6s9m3.weaviate.network"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "imlIdfNFZ8eC6rm4iCvSrTJyQeoBo6KjQtLT"))
) as client:
    print(client.is_ready())


    images = ["./deer.jpg"]
    names = ["deer.jpg"]

    def img_embedder(images, names):
        client.connect()

        batch_size = 16
        image_arr = None

        for i in tqdm(range(0, len(images), batch_size)):
            # select batch of images
            batch = []
            for j in range(i, min(i + batch_size, len(images))):
                img = Image.open(images[j]).convert("RGB")
                batch.append(img)
            # process and resize
            batch = processor(
                text=None,
                images=batch,
                return_tensors='pt',
                padding=True
            )['pixel_values'].to(device)
            # get image embeddings
            batch_emb = model.get_image_features(pixel_values=batch)
            # convert to numpy array
            batch_emb = batch_emb.squeeze(0)
            batch_emb = batch_emb.cpu().detach().numpy().tolist()
            # add to larger array of all image embeddings
            if image_arr is None:
                image_arr = batch_emb
            else:
                image_arr += [batch_emb]

        videoSearch = client.collections.get("VideoSearch")
        for i in zip(image_arr, names):
            uuid = videoSearch.data.insert(
                {
                    "name": i[1]
                },
                vector=image_arr
            )
            print(uuid)
        print("Completed")

# img_embedder(images, names)