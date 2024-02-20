import streamlit as st
import pandas as pd
from io import StringIO
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
import os
import tempfile
import tabulate

os.environ['OPENAI_API_KEY'] = "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"
client = OpenAI()

#streamlit part##############################################################################

st.title("AI Question Answering App")

# File upload section
st.header("Upload CSV Files")
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Context input
st.header("Provide Context")
context = st.text_area("Enter context")

# Question input
st.header("Ask a Question")
question = st.text_input("Ask your question")

##############################################################################################

llm=ChatOpenAI(temperature=0.5)

# Add a button for running the agent
if st.button('Run'):
    # Check if files are uploaded
    if uploaded_files:
        # Create a list to hold the paths to the temporary files
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                # Write the contents of the uploaded file to the temporary file
                f.write(uploaded_file.getvalue())
                temp_file_paths.append(f.name)

        # Create the agent with the list of temporary file paths
        agent_executer = create_csv_agent(llm, temp_file_paths, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
        response = agent_executer.invoke(context + question)
        st.write(response['output'])
    else:
        st.write("Please upload a file.")