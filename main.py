import streamlit as st
import pandas as pd
import json
import re

# Set up the page configuration
st.set_page_config(layout="wide", page_title="NL Data Explorer", page_icon="📈")

# --- Session State Management ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'original_df' not in st.session_state:
    st.session_state.original_df = pd.DataFrame()
if 'operation_history' not in st.session_state:
    st.session_state.operation_history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'table'  # 'table', 'chart'
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'bar'
if 'last_command' not in st.session_state:
    st.session_state.last_command = ""
if 'suggestion_options' not in st.session_state:
    st.session_state.suggestion_options = []

# --- Helper Functions (Mocking the NLP/Backend Logic) ---
def parse_command(command, df):
    """
    Mocks a natural language parsing function.
    It returns a list of possible operations based on the command.
    Each operation is a dictionary with 'operation', 'description', and 'params'.
    """
    command = command.lower()
    options = []
    
    # Example 1: 'show seasonality by region' -> Group by region, count/sum over time
    if any(word in command for word in ['seasonality', 'by region', 'time series']):
        if 'region' in df.columns and 'date' in df.columns:
            options.append({
                'operation': 'pivot_table',
                'description': 'Show seasonality of sales by region',
                'params': {
                    'index': 'date', 
                    'columns': 'region', 
                    'values': 'sales', 
                    'aggfunc': 'sum'
                }
            })
            options.append({
                'operation': 'group_by_count',
                'description': 'Group by region and month to show count of items sold',
                'params': {
                    'group_col': 'region', 
                    'date_col': 'date'
                }
            })
        else:
            options.append({
                'operation': 'info_text',
                'description': "Could not find 'region' and 'date' columns to show seasonality.",
                'params': {}
            })
    
    # Example 2: 'top 5 products this quarter' -> Filter by date, sort, head
    elif any(word in command for word in ['top 5', 'best sellers', 'top products']):
        if 'product' in df.columns and 'sales' in df.columns:
            options.append({
                'operation': 'top_n',
                'description': 'Show top 5 products by sales',
                'params': {
                    'n': 5,
                    'by_column': 'sales',
                    'sort_column': 'product'
                }
            })
            options.append({
                'operation': 'top_n',
                'description': 'Show top 5 products by number of unique customers',
                'params': {
                    'n': 5,
                    'by_column': 'customer_id',
                    'sort_column': 'product',
                    'agg_func': 'nunique'
                }
            })
    
    # Example 3: Simple filters and sorts
    elif 'filter' in command or 'sort' in command or 'show only' in command:
        col_match = re.search(r'by (\w+)', command)
        val_match = re.search(r'is (\w+)', command)
        
        if col_match and val_match and col_match.group(1) in df.columns:
            options.append({
                'operation': 'filter',
                'description': f"Filter where {col_match.group(1)} is {val_match.group(1)}",
                'params': {
                    'column': col_match.group(1),
                    'value': val_match.group(1)
                }
            })
    
    # Default/Vague command
    if not options:
        options.append({
            'operation': 'info_text',
            'description': "Sorry, I'm not sure what you mean. Please try a different command.",
            'params': {}
        })

    return options

def apply_operation(df, operation_dict):
    """Applies a single operation and returns the modified DataFrame."""
    op = operation_dict['operation']
    params = operation_dict['params']
    
    if op == 'pivot_table':
        df = pd.pivot_table(df, **params).reset_index()
    elif op == 'top_n':
        agg_func = params.get('agg_func', 'sum')
        if agg_func == 'sum':
            df = df.groupby(params['sort_column']).agg({params['by_column']: 'sum'}).nlargest(params['n'], params['by_column']).reset_index()
        elif agg_func == 'nunique':
            df = df.groupby(params['sort_column']).agg({params['by_column']: 'nunique'}).nlargest(params['n'], params['by_column']).reset_index()
    elif op == 'filter':
        df = df[df[params['column']] == params['value']]
    elif op == 'group_by_count':
        df['month'] = pd.to_datetime(df[params['date_col']]).dt.to_period('M')
        df = df.groupby([params['group_col'], 'month']).size().reset_index(name='count')
        df['month'] = df['month'].astype(str)
    
    return df

# --- UI Components ---
st.title("📊 NL Data Explorer")
st.markdown("Talk, don't tool. Describe your needs in plain language and get a useful view.")

# --- Sidebar for File Upload and History ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            st.session_state.original_df = pd.read_csv(uploaded_file)
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.operation_history = []
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.header("Operation History")
    for i, op in enumerate(st.session_state.operation_history):
        st.markdown(f"**Step {i+1}:** {op['description']}")

# --- Main Content Area ---
if not st.session_state.df.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        # --- Command Box ---
        st.subheader("What do you want to see?")
        command = st.text_input("Enter your command here...", value=st.session_state.last_command)
        
        if st.button("Apply Command"):
            if command:
                st.session_state.last_command = command
                st.session_state.suggestion_options = parse_command(command, st.session_state.original_df)
            else:
                st.warning("Please enter a command.")
        
        # --- Suggestions Panel ---
        if st.session_state.suggestion_options:
            st.markdown("---")
            st.subheader("I have a few ideas...")
            st.write("Please select the one that best matches your intent:")
            
            for i, option in enumerate(st.session_state.suggestion_options):
                if st.button(f"Option {i+1}: {option['description']}", key=f"suggestion_btn_{i}"):
                    # Apply the chosen operation
                    st.session_state.df = apply_operation(st.session_state.original_df.copy(), option)
                    st.session_state.operation_history.append(option)
                    st.session_state.suggestion_options = [] # Clear suggestions
                    st.success(f"Applied: {option['description']}")
                    
                    # Heuristically decide on view type
                    if 'pivot' in option['operation'] or 'group' in option['operation']:
                        st.session_state.current_view = 'chart'
                        st.session_state.chart_type = 'bar'
                    else:
                        st.session_state.current_view = 'table'

    with col2:
        # --- Operation Explain Panel ---
        st.subheader("Current View Explained")
        if st.session_state.operation_history:
            last_op = st.session_state.operation_history[-1]
            st.markdown(f"**You are currently viewing data based on this command:**")
            st.markdown(f"**Description:** `{last_op['description']}`")
            st.json(last_op['params'])
        else:
            st.info("No operations applied yet. This is a preview of the raw data.")
    
    st.markdown("---")
    
    # --- Chart/Table Area and Export ---
    st.subheader("Data View")
    
    if st.session_state.current_view == 'table':
        st.dataframe(st.session_state.df)
    else:
        # Simple chart generation based on the current DataFrame
        if len(st.session_state.df.columns) >= 2:
            st.subheader("Chart View")
            x_col = st.session_state.df.columns[0]
            y_col = st.session_state.df.columns[1]
            
            chart_options = ['bar chart', 'line chart', 'area chart']
            selected_chart = st.selectbox("Choose a chart type:", chart_options)
            
            if selected_chart == 'bar chart':
                st.bar_chart(st.session_state.df, x=x_col, y=y_col)
            elif selected_chart == 'line chart':
                st.line_chart(st.session_state.df, x=x_col, y=y_col)
            elif selected_chart == 'area chart':
                st.area_chart(st.session_state.df, x=x_col, y=y_col)
    
    # --- Export Button ---
    st.markdown("---")
    st.subheader("Export")
    csv_export = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Current View as CSV",
        data=csv_export,
        file_name='data_explorer_export.csv',
        mime='text/csv',
    )

else:
    st.info("Please upload a CSV file to begin.")