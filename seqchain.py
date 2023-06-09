import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv())  # read local .env file

llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)

third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)

fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following"
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)


# Chaining Multiple chains.
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "language", "followup_message"],
    verbose=True
)

# Review for 300 words
print(overall_chain("I love this product. It is the best product I have ever used. I will definitely buy it again."))

"""
result

{
    "Review": "I love this product. It is the best product I have ever used. I will definitely buy it again.",
    "English_Review": "Me encanta este producto. Es el mejor producto que he utilizado. Sin duda lo compraré de nuevo.",
    "summary": "The reviewer loves the product and thinks it's the best they've ever used, and will definitely buy it again.",
    "language": "English.",
    "followup_message": "Thank you for taking the time to review our product. We are thrilled to hear that you love it and that it is the best you've ever used. We truly appreciate your positive feedback and we look forward to serving you again in the future. Thank you for choosing our product!"
}
"""