import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os

# Page configuration
st.set_page_config(page_title="Finance Variance Analysis", layout="wide")

st.title("Finance Variance Analysis")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    
    # Try to get API key from Streamlit secrets first (for deployment)
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key loaded from secrets")
    except (KeyError, FileNotFoundError):
        # Fall back to user input for local development
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
        if not api_key:
            st.warning("Please enter your Google Gemini API Key to proceed.")

# Data Source
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("Using sample data. Upload a CSV to use your own data.")
    except FileNotFoundError:
        st.error("sample_data.csv not found. Please upload a file.")
        st.stop()

# Display Data Preview
with st.expander("Data Preview"):
    st.subheader("Complete Dataset Overview")
    
    # Create pivot tables for Plan and Actuals
    try:
        # Pivot for Plan values
        plan_pivot = df.pivot(index='Account', columns='Month', values='Plan')
        
        # Pivot for Actuals values  
        actual_pivot = df.pivot(index='Account', columns='Month', values='Actuals')
        
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Reorder columns to match month order (only if all months exist)
        available_months = [m for m in month_order if m in plan_pivot.columns]
        plan_pivot = plan_pivot[available_months]
        actual_pivot = actual_pivot[available_months]
        
        # Display Plan section
        st.write("### Plan")
        st.dataframe(plan_pivot, use_container_width=True)
        
        # Display Actual section
        st.write("### Actual")
        st.dataframe(actual_pivot, use_container_width=True)
        
        # Display summary statistics
        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Account Types", len(df['Account'].unique()))
        with col3:
            st.metric("Months", len(available_months))
            
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        st.write("**Raw Data Preview (first 20 rows):**")
        st.dataframe(df.head(20))

# Context Area
default_context = ""
if uploaded_file is None:
    default_context = """REVENUE VARIANCES:
Q1: Jan favorable (+$2k) due to unexpected contract renewal. Feb unfavorable (-$2k) from delayed customer payment. Mar favorable (+$5k) from new client onboarding ahead of schedule.
Q2: Apr favorable (+$2k) from seasonal uptick. May unfavorable (-$2k) due to service disruption. Jun favorable (+$5k) from promotional campaign success.
Q3: Jul unfavorable (-$2k) from summer slowdown. Aug favorable (+$2k) from new product launch. Sep favorable (+$1k) from stable operations.
Q4: Oct unfavorable (-$2k) from competitive pricing pressure. Nov favorable (+$5k) from holiday season boost. Dec favorable (+$2k) from year-end contracts.

COGS VARIANCES:
Q1-Q2: Minor variances ($1-2k) due to normal supplier price fluctuations and volume discounts.
Q3-Q4: Steady at plan or slightly favorable, reflecting improved procurement efficiency.

LABOR VARIANCES:
Generally at plan. Mar, May, Sep, and Nov show +$500-1k variances due to overtime for project deadlines and temporary staffing needs during peak periods.

FIXED COSTS VARIANCES:
Minor monthly fluctuations (+/-$200-500) from utilities rate changes, property tax adjustments, and insurance premium updates. Overall trending slightly above plan due to cost inflation.

VARIABLE COSTS VARIANCES:
Track closely with revenue patterns. Favorable when revenue is down (Feb, May, Jul, Oct) and unfavorable when revenue exceeds plan, reflecting commission structures and shipping costs.

G&A VARIANCES:
Q1-Q2: Minor variances (+/-$200-300) from office supplies, travel costs, and professional services timing.
Q3-Q4: Slightly higher activity from year-end audit preparations and strategic planning initiatives.

DEPRECIATION:
Consistently at plan. No variance as depreciation follows straight-line schedule.

INTEREST EXPENSE:
May-Dec show favorable variances ($100-300) due to early principal payments reducing outstanding debt balance and lowering interest charges.

OPERATING TAXES:
Minor variances (+/-$20-70) from quarterly true-ups based on actual revenue performance and regulatory adjustments.

INCOME TAXES:
Variances (+/-$100-250) align with revenue and profitability fluctuations. Higher revenue months show higher tax variance."""

context = st.text_area("Variance Context", value=default_context, height=300)

# Chat Interface
st.header("Ask Questions")
question = st.text_input("Ask a question about your data:")

if question and api_key:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0)
        
        # Create pandas dataframe agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True  # Required for pandas operations
        )
        
        # Combine context and question
        final_prompt = f"""Context provided by user: {context}

Question: {question}

Please provide a clear and concise answer based on the data and context provided."""
        
        with st.spinner("Analyzing..."):
            response = agent.invoke(final_prompt)
            
            # Display the response
            if isinstance(response, dict) and 'output' in response:
                st.write(response['output'])
            else:
                st.write(response)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif question and not api_key:
    st.error("Please enter your API Key in the sidebar.")
