from groq import Groq #type: ignore
import streamlit as st # type: ignore
from data_insights import generate_summary

# Initialize Groq API
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
def process_query(query, data_summary):
    system_message = (
        f"You are a highly skilled and professional data analyst proficient in Python programming and data visualization tools like Pandas and Matplotlib.\n"
        f"Below is the summary of a dataset:\n{data_summary}\n"
        "Your task is to provide clean and well-structured Python code that fulfills the following requirements:\n"
        "1. The code must address the user's question or request explicitly. The question is: '{query}'.\n"
        "2. Assume the dataset has already been loaded into a DataFrame named `df`.\n"
        "3. If the query involves generating a visualization (e.g., bar chart, line chart, etc.), ensure that:\n"
        "   a. The chart is well-labeled with appropriate titles, axis labels, and legends (if applicable).\n"
        "   b. The chart displays no more than the top or bottom 10 categories based on count or value unless the user explicitly requests to show all categories.\n"
        "   c. Include a brief annotation or highlight specific features of the visualization where necessary.\n"
        "4. If numerical analysis is requested, provide clear and concise insights such as statistical summaries, correlations, or trends.\n"
        "5. Include comments in the code to explain each major step for better readability.\n"
        "6. The output should contain only the Python code (no explanations or extraneous text).\n"
    )

    # Chat messages for the LLM
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    # Query Llama
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )

    # Return generated code
    return response.choices[0].message.content
