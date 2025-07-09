from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
name=llm("What's my name?")
print(name)

prompt=PromptTemplate(
    input_variables=["name"],
    template="What is a good name for a company that makes {name}?",
)
prompt.format(name="socks")

prompt_template_name=PromptTemplate(
    input_variables=["cuisine"],
    template="What are some good food items from {cuisine} cuisine?",
)
name_chain = LLMChain(llm=llm, prompt=prompt,output_key="resturant_name")

prompt_template_items=PromptTemplate(
    input_variables=["cuisine"],
    template="What are some good food items from {cuisine} cuisine?",
)
food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items,output_key="food_items")

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")

from langchain.chains import SimpleSequentialChain
chain= SimpleSequentialChain(chains=[name_chain, food_items_chain])
response=chain.run("Indian")
print(response)

from langchain.chains import SequentialChain
chain=SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=["cuisine"],
    output_variables=["resturant_name", "food_items"],
    verbose=True
)

chain.run({'cuisine': 'Indian'})