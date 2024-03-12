import streamlit as st
import os
from langchain.llms import OpenAI
from constants import openai_key
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"]=openai_key

# Streamlit code
def main():
    st.title("Restaurant Name and Menu Suggestion")
    
    # Input field for cuisine
    cuisine = st.text_input("Enter the cuisine: (e.g., Indian)")
    
    if st.button("Suggest"):
        st.write("Suggested Restaurant Name:")
        restaurant_name = suggest_restaurant_name(cuisine)
        st.write(restaurant_name)

        st.write("Suggested Food Menu:")
        food_menu = suggest_food_menu(restaurant_name)
        st.write(food_menu)

    if st.button("Suggest Sequentially"):
        result = run_sequential_chain(cuisine)
        st.write("Suggested Restaurant Name:", result['restaurant_name'])
        st.write("Suggested Food Menu:", result['menu_items'])

# Function to suggest a restaurant name based on cuisine
def suggest_restaurant_name(cuisine):
    llm = OpenAI(temperature=0.6)
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template='Suggest me a good name for my restaurant with {cuisine} food')
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    return name_chain({'cuisine': cuisine})

# Function to suggest food menu for a restaurant
def suggest_food_menu(restaurant_name):
    llm = OpenAI(temperature=0.6)
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest me a good food menu for {restaurant_name}. Return it as comma separated list')
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    return food_items_chain({'restaurant_name': restaurant_name})

# Function to run the sequential chain
def run_sequential_chain(cuisine):
    llm_name = OpenAI(temperature=0.7)
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template='Suggest me a good name for my restaurant with {cuisine} food')
    name_chain = LLMChain(llm=llm_name, prompt=prompt_template_name, output_key="restaurant_name")

    llm_items = OpenAI(temperature=0.7)
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest me a good food menu for {restaurant_name}. Return it as comma separated list')
    food_items_chain = LLMChain(llm=llm_items, prompt=prompt_template_items, output_key="menu_items")

    sequential_chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    return sequential_chain({'cuisine': cuisine})

if __name__ == "__main__":
    main()
