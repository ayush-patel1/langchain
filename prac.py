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


from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")