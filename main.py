import openai
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from jproperties import Properties

# setting up properties
configs = Properties()

with open('config.properties', 'rb') as config_file:
    configs.load(config_file)

# Loading the secret key of openAI from config.properties
secretKey = configs.get('SECRET_KEY').data

# Creating llm model
# Note: Controls the randomness of the model's output. A higher value makes the output more random, while a lower value makes it more deterministic. You can set this parameter in the range [0.0, 1.0].
llm = OpenAI(openai_api_key= secretKey, temperature=0.5)

# Testing by giving a prompt
# Prompt 1

prompt1 = "What is the capital of India?"

# print(llm_chain.invoke(prompt1))
answer1 = llm.predict(prompt1)
print(f"Prompt 1: {prompt1}")
print(f"Answer: {answer1}")


# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate.from_template(template)

# llm_chain = prompt | llm





