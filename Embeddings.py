import os
import tiktoken
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
from utilities.azureblobstorage import get_all_files, download_blob
from utilities.utils import colorprint, project_2D,string2float
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


try:
    df = pd.read_csv('data/df.csv',index_col=False)
    brand_df=pd.read_csv('data/brand_df.csv',index_col=False)
    colorprint('Found df in file','50')
except:
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
    df.to_csv('data/df.csv',index=False)

    brand_df  = df.drop_duplicates(subset=['Brand'],inplace=False)[['Business','Brand']]
    brand_df.to_csv('data/brand_df.csv',index=False)
print(df.head())
print(brand_df.head())

try:
    df=pd.read_csv('data/df_embedded.csv',index_col=False)
    embeddings=df.embedding.apply(lambda x: x.strip("[]").replace("'","").split(", ") )
    embeddings=[string2float(x) for x in embeddings]
    df.embedding = embeddings
    
    brand_df=pd.read_csv('data/brand_df_embedded.csv',index_col=False)
    embeddings=brand_df.embedding.apply(lambda x: x.strip("[]").replace("'","").split(", ") )
    embeddings=[string2float(x) for x in embeddings]
    brand_df.embedding = embeddings

    colorprint('Found embeddings in a file ', '50')
except:
    colorprint("Calculating embeddings")
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
    def get_embedding(text) -> list[float]:
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=text, engine=EMBEDDING_MODEL)["data"][0]["embedding"]
    df = df.assign(embedding=df['Keyword'].apply(lambda x: get_embedding(x)))
    colorprint('Calculated embeddings for Keywords')
    print(df.head())

    brand_df = brand_df.assign(embedding=brand_df['Brand'].apply(lambda x: get_embedding(x)))
    colorprint('Calculated embeddings for Brands')
    print(brand_df.head())

    df.to_csv('data/df_embedded.csv',index=False)
    brand_df.to_csv('data/brand_df_embedded.csv',index=False)

colorprint('adding clusters')

colorprint('Calculating clusters')

keylen=df.shape[0]
brandlen=brand_df.shape[0]
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(df['embedding'].to_list()+brand_df['embedding'].tolist())
df = df.assign(cluster=kmeans.labels_[:keylen])
print(df.head())

brand_df = brand_df.assign(cluster=kmeans.labels_[keylen:])

print(brand_df.head())

colorprint('Projecting the embeddings to 2D for plotting')



tsne = TSNE(
    n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
)

matrix1 = np.vstack(df.embedding.values)
keyword_len=matrix1.shape[0]
print(matrix1.shape)

matrix2 = np.vstack(brand_df.embedding.values)
brand_len=matrix2.shape[0]
print(matrix2.shape)


all_matrix= np.concatenate((matrix1,matrix2))
print(type(all_matrix))
print(all_matrix.shape)

vis_dims2 = tsne.fit_transform(all_matrix)
x = [x for x, y in vis_dims2[:keyword_len]]
y = [y for x, y in vis_dims2[:keyword_len]]
brand_x = [x for x, y in vis_dims2[keyword_len:]]
brand_y = [y for x, y in vis_dims2[keyword_len:]]



for category, color in enumerate(["purple", "green", "red"]):
    xs = np.array(x)[df.cluster == category]
    ys = np.array(y)[df.cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()
    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    brand_xs=np.array(brand_x)[brand_df.cluster==category]
    brand_ys=np.array(brand_y)[brand_df.cluster==category]
    plt.scatter(brand_xs, brand_ys, color=color, alpha=1)


plt.show()
print(df.columns)
print(brand_df[["Brand","cluster"]])

