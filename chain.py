import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv())  # read local .env file

llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that make {product}?"
)

chain_one = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company name: {company_name}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

# result of first output is pass as a argument to second input
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

product = "Queen Size Sheet Set"

print(overall_simple_chain.run(product))

"""
result

"Royal Linens" would be a suitable name for a company that makes Queen Size Sheet Sets. It conveys a sense of luxury and high-quality bedding fit for royalty.
Royal Linens makes high-quality Queen Size Sheet Sets fit for royalty, conveying a sense of luxury and elegance.
"""
