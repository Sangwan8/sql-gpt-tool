import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="GPT-Powered SQL Generator", page_icon="🧠")
st.title("🧠 GPT-Powered SQL Generator")

st.markdown("### Step 1: Upload a CSV File (optional)")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None
table_info = ""

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Extract schema info
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).tolist()
    table_info = "\n".join(f"{col} ({dtype})" for col, dtype in zip(columns, dtypes))

    st.markdown("#### Detected Schema:")
    st.code(table_info)

# Prompt section
st.markdown("### Step 2: Describe the SQL you want to generate")
user_prompt = st.text_input("For example: 'Find countries with most missing DOB'")

if st.button("Generate SQL") and user_prompt:
    with st.spinner("Thinking like an analyst..."):
        # Build system prompt based on context
        base_system_prompt = "You are a helpful SQL analyst. Only return valid SQL queries without explanation."
        if table_info:
            base_system_prompt += f"\n\nHere is the table schema:\n{table_info}"

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        sql_output = response.choices[0].message.content
        st.success("✅ Copy and paste this SQL into SSMS or Power BI:")
        st.code(sql_output, language="sql")
