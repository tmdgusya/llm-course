import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator

_ = load_dotenv(find_dotenv())  # read local .env file

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
).from_loaders([loader])

query = "Please list all your shirts with sun protection \ in a table in markdown and summurize each one"

response = index.query(query)

display(Markdown(response))