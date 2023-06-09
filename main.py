import os
import openai

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model="gpt-3.5-turbo"):
    message = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        temperature=0,
    )
    return response.choices[0].message["content"]


# print(get_completion("What is 1+1?"))

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks
into a style that is {customer_style}.
text: ```{customer_email}```
"""

# print(get_completion(prompt))

chat = ChatOpenAI(temperature=0.0)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt)  # show template and variables
print(prompt_template.messages[0].prompt.input_variables)  # show variables

customer_messages = prompt_template.format_messages(
    style=customer_style,
    text=customer_email,
)

print(customer_messages[0])

# customer_response = chat(customer_messages)
# print(customer_response.content)

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

# reuse prompt template
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply
)

print(service_messages[0].content)

service_response = chat(service_messages)
print(service_response.content)