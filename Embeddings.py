import os
import tiktoken
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
from utilities.azureblobstorage import get_all_files, download_blob
from utilities.utils import colorprint
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define embedding model and encoding
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDINGS_ENGINE_DOC") #'text-embedding-ada-002'
COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL")
encoding = tiktoken.get_encoding('cl100k_base')

# Defime files access
account_name = os.environ['BLOB_ACCOUNT_NAME']
account_key = os.environ['BLOB_ACCOUNT_KEY']
connect_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
container_name = os.environ['BLOB_CONTAINER_NAME']

# Get the data
files_data = get_all_files()
print(files_data)
files_data = list(map(lambda x: {'filename': x['filename']}, files_data))
print(files_data)
allfiles = []
for fd in files_data:
        allfiles.append(fd['filename'])


file_name = allfiles[0]
with open(f"./data/{file_name}", "wb") as my_downloaded_blob:
    blob_data = download_blob(file_name)
    blob_data.readinto(my_downloaded_blob)


df= pd.read_excel(os.path.join('data',file_name))[['Business','Brand','Keyword']]
print(df)
df = df.assign(keyword_token_count=df['Keyword'].apply(lambda x: len(encoding.encode(x))))

print(df.head())
total_tokens = df['keyword_token_count'].sum()

cost_for_embeddings = total_tokens / 1000 * 0.0004
print(f"Test would cost ${cost_for_embeddings} for embeddings")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def get_embedding(text) -> list[float]:
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, engine=EMBEDDING_MODEL)["data"][0]["embedding"]

df = df.assign(embedding=df['Keyword'].apply(lambda x: get_embedding(x)))
print(df.head())

#openai.Embedding.create('ontharingscreme intieme delen', engine=EMBEDDING_MODEL)

# train k-means on df embeddings

colorprint('adding clusters')
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(df['embedding'].to_list())
df = df.assign(cluster=kmeans.labels_)
print(kmeans.labels_)
print(df.head())

#Now that we have a cluster per row, let's use t-SNE to project our embeddings into 2d space and visualize the clusters.


tsne = TSNE(
    n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
)

matrix = np.vstack(df.embedding.values)
print(matrix.shape)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

print(x[0:4])
print(y[0:4])


for category, color in enumerate(["purple", "green", "red"]):
    xs = np.array(x)[df.cluster == category]
    ys = np.array(y)[df.cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.show()
