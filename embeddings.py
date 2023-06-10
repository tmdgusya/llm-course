import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# print(docs) print all documents in the csv file

print(docs[0])

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0.0)

# join all the documents contents into one string
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

# provide all the content with query to GPT
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")

# result of this response is same as before we've seen when we used similarity_search
# This is how the embeddings works
display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

query = "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

# result of this response is same as before we've seen when we used similarity_search
response = qa_stuff.run(query)


