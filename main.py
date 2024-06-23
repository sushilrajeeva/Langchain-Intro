import warnings
# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import openai
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from jproperties import Properties
from operator import itemgetter



from langchain.agents import AgentType, initialize_agent, load_tools



# setting up properties
configs = Properties()

with open('config.properties', 'rb') as config_file:
    configs.load(config_file)

# Loading the secret key of openAI from config.properties
secretKey = configs.get('SECRET_KEY').data

# Creating llm model
# Note: Controls the randomness of the model's output. A higher value makes the output more random, while a lower value makes it more deterministic. You can set this parameter in the range [0.0, 1.0].
llm = OpenAI(openai_api_key= secretKey, temperature=0.5)

# Creating a function to display question and answer
counter = 0
def display(question: str, answer: str) -> None:
    global counter
    counter += 1
    print(f"Question {counter}: {question}")
    print(f"Answer: {answer}")
    print()
    print("****************************************")
    print()

# Testing by giving a prompt
# Prompt 1
# prompt1 = "What is the capital of India?"
# answer1 = llm.predict(prompt1)
# print(f"Prompt 1: {prompt1}")
# print(f"Answer: {answer1}")

## Using llm chain and prompt template

template = """Question:{question}

Answer: .(no sentence just answer)"""

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llm
question1 = "What is the capital of India?"
answer1 = llm_chain.invoke(question1).strip()
display(question1, answer1)

question2 = "What is the capital of United States of America?"
answer2 = llm_chain.invoke(question2).strip()
display(question2, answer2)


# Using agents to feed data to llm 
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
question3 = "How old will Virat Kohli be in 2030?"
answer3 = agent.run(question3)
display(question3, answer3)
question4 = "Who won the 2024 loksabha election in india?"
answer4 = agent.run(question4)
display(question4, answer4)

