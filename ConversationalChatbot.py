import streamlit as st
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from constants import openai_key
# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"]=openai_key
# Set up Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Chat with Tejaswini's AI Buddy")

# Initialize chat model
chat = ChatOpenAI(temperature=0.5)

# Initialize flow messages
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are chatting with Tejaswini's AI Buddy")
    ]

# Function to get chat model response
def get_chatmodel_response(question):
    # Append user message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['flowmessages'].append(HumanMessage(content=question, timestamp=timestamp))
    # Get response from chat model
    answer = chat(st.session_state['flowmessages'])
    # Append AI message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['flowmessages'].append(AIMessage(content=answer.content, timestamp=timestamp))
    return answer.content

# Streamlit UI components
input_text = st.text_input("Input: ", key="input")
submit_button = st.button("Ask the question")

# If submit button is clicked
if submit_button:
    response = get_chatmodel_response(input_text)
    # Display response with timestamp
    st.subheader("The Response is")
    st.write(response)
