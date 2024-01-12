import pandas as pd
import tiktoken
from openai import OpenAI
#from embeddings_utils import * 
#from openai.embeddings_utils import get_embedding 
from dotenv import dotenv_values

# read csv and check the first few entires
anime = pd.read_csv('./data/anime_with_synopsis.csv')
print(anime.head())
print("\n")

# combine textual information
anime['combined_info'] = anime.apply(lambda row: f"Title: {row['Name']}, Overview: {row['sypnopsis']}, Genres: {row['Genres']}", axis=1)
print(anime.head(2))
print("\n")

# count the number of tokens for combined_info, ensuring
# they do not exceed OpenAI embedding 8191 token limit
embedding_encoding = "cl100k_base" # encoding for text-embedding-ada-002
max_tokens = 8000

encoding = tiktoken.get_encoding(embedding_encoding)
print("Encoding:")
print(encoding)
print("\n")

# omit descriptions that are too long to embed
anime["n_tokens"] = anime.combined_info.apply(lambda x: len(encoding.encode(x)))
anime = anime[anime.n_tokens <= max_tokens]
print(f"Anime size = {anime.size}")
print("Ada combined info:")
print(anime.combined_info)
print("\n")

# embedding and vectorDB
config = dotenv_values(".env")
api_key = config["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
embedding_model = "text-embedding-ada-002"

"""
anime["embedding"] = anime.combined_info.apply(lambda x: get_embedding(x, engine=embedding_model))
print("DONE - HEAD")
print(anime.head(2))
"""

def get_embeddings(row, embedding_model):
    print("Row:")
    print(row)
    return client.embeddings.create(
        model=embedding_model,
        input=row
    )
    #data = response.data[0].embedding

sliced_data = anime[:5]
print(anime.head())

"""
print(sliced_data.combined_info)
print("\n")

embedded_data = sliced_data.combined_info.apply(lambda x: get_embeddings(x, embedding_model)) 
print(embedded_data)
print("\n")
"""
