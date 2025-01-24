import streamlit as st  # type: ignore
import pandas as pd
from data_insights import generate_summary
from query_processor import process_query
from code_executor import run_generated_code
import warnings

warnings.filterwarnings("ignore")

# Streamlit App UI
st.title("Data Navigator: Simplify Exploration")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file to start exploring your data:", type=["csv"])

if uploaded_file:
    # Read dataset
    dataset = pd.read_csv(uploaded_file)
    
    # Generate data summary
    data_summary = generate_summary(dataset)
    
    # Display dataset snapshot
    st.write("### Dataset Snapshot:")
    st.dataframe(dataset.head())

    # # Display data summary
    # if st.checkbox("View Dataset Summary"):
    #     st.write("### Dataset Summary:")
    #     st.text(data_summary)

    # Detailed data insights
    if st.checkbox("View Detailed Data Insights"):
        tabs = st.tabs(["Null Values", "Unique Values", "Duplicate Records", "Descriptive Stats", "Numeric Summary"])

        # Null values
        with tabs[0]:
            null_counts = dataset.isnull().sum()
            st.write("Null Values by Column:")
            st.dataframe(null_counts)
            st.write(f"**Total Null Values:** {null_counts.sum()}")

        # Unique values
        with tabs[1]:
            unique_counts = {col: dataset[col].nunique() for col in dataset.columns}
            st.write("Unique Values by Column:")
            st.dataframe(pd.DataFrame(list(unique_counts.items()), columns=["Column", "Unique Count"]))

        # Duplicate records
        with tabs[2]:
            duplicate_count = dataset.duplicated().sum()
            st.write(f"**Total Duplicate Records:** {duplicate_count}")

        # Descriptive statistics
        with tabs[3]:
            st.write("Statistical Summary:")
            st.dataframe(dataset.describe())

        # Numeric column summary
        with tabs[4]:
            numeric_columns = dataset.select_dtypes(include="number")
            if not numeric_columns.empty:
                st.write("Numeric Column Summary:")
                numeric_summary = numeric_columns.describe().T
                st.dataframe(numeric_summary)
            else:
                st.write("No numeric columns in the dataset.")

    # Query input for data exploration
    st.subheader("Ask Questions About Your Data")
    query = st.text_input("Enter a question about the data:")

    if query:
        with st.spinner("Analyzing your query..."):
            try:
                # Generate code from the query
                generated_code = process_query(query, data_summary)

                if st.checkbox("Display Generated Code"):
                    st.code(generated_code, language="python")

                # Execute the generated code
                execution_results, execution_output = run_generated_code(generated_code, dataset)

                # Display results
                if execution_output:
                    st.write(execution_output)
                
                # Handle matplotlib plots
                if "plt" in execution_results:
                    st.pyplot(execution_results["plt"].gcf())  # type: ignore
                
                # Handle general execution results
                if isinstance(execution_results, str):
                    st.error(execution_results)
                else:
                    st.success("Code executed successfully!")
            except Exception as execution_error:
                st.error(f"An error occurred during code execution: {execution_error}")
