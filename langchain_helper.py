from langchain.llms import Gemini
from langchain.prompts import PromptTemplate
from secret_key import openai_api_key
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
import os
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = OpenAI(temperature=0.7)
name=llm("What's founder of google?")
print(name)


def generate_resturant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="What is a good name for a restaurant that serves {cuisine} cuisine?",
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="resturant_name")            
    prompt_template_items = PromptTemplate(
        input_variables=["cuisine"],
        template="What are some good food items from {cuisine} cuisine?",
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items
        , output_key="menu_items")  
    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["resturant_name", "menu_items"],
        verbose=True
    )
    response = chain.run({'cuisine': cuisine})
    print(response)  
    return{  
        'resturant_name': 'curry delight',
        'menu_items': ['curry', 'naan', 'biryani']
    }