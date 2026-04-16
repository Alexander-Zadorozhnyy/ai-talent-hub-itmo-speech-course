import pandas as pd
from tqdm import tqdm

# https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News/tree/main
df = pd.read_parquet("bloomberg_financial_data.parquet.gzip", engine="fastparquet")

# Extract the "Article" column and write to text file
with open("bloomberg_financial_data.txt", "w", encoding="utf-8") as f:
    for article in tqdm(df["Article"]):
        # Convert to string and write each article as a line
        f.write(str(article) + "\n")
