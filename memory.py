import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory


_ = load_dotenv(find_dotenv())  # read local .env file


# llm = ChatOpenAI(temperature=0.0)
# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#    llm=llm,
#    memory = memory,
#    verbose=True
#)

# memory.save_context({"input": "Hi"}, {"output": "My name is roach"}) it doesn't work
# print(conversation.predict(input="What is my name?"))

# print(memory.buffer)

# print(memory.load_memory_variables({}))

#############################################
# memory = ConversationBufferWindowMemory(k=1) # it means saving only the last message

# memory.save_context({"input": "Hi"}, {"output": "My name is roach"})
# memory.save_context({"input": "What is my name?"}, {"output": "Your name is roach"})
# print(memory.load_memory_variables({})) # {'history': 'Human: What is my name?\nAI: Your name is roach'}

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)
# conversation.predict(input="Hi, my name is roach")
# conversation.predict(input="What is 1+1?")
# conversation.predict(input="What is my name?") # A.I doesn't know the answer. becuase, it only remember last one message


from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0)

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})
print(memory.load_memory_variables({}))

"""
Result of load_memory_variables function
{'history': 'System: The human and AI engage in small talk before the human asks about their schedule for the day. 
The AI informs the human of a meeting with their product team at 8am, 
time to work on their LangChain project from 9am-12pm, 
and a lunch meeting with a customer interested in the latest in AI.'}
"""

print(conversation.predict(input="What would be a good demo to show?"))

"""
Well, that depends on the audience and the purpose of the demo. 
If you're trying to showcase a new product, 
you could create a demo that highlights its key features and benefits. 
If you're presenting to investors, you might want to focus on the market potential and revenue projections.
 Alternatively, if you're trying to educate people about a complex topic, 
 you could create an interactive demo that breaks down the information into digestible chunks. 
 Ultimately, the best demo is one that engages your audience and effectively communicates your message. 
 Do you have any specific goals or requirements for your demo?
"""

print(memory.load_memory_variables({}))
