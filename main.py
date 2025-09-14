import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import os

# Try to use Groq first, fallback to Gemini if needed
def get_ai_client():
    if "GROQ_API_KEY" in st.secrets:
        return "groq"
    elif "GEMINI_API_KEY" in st.secrets:
        return "gemini"
    else:
        return "local"  # Use local fallback when no API keys are available


def filter_dataframe(column: str, operator: str, value: str):
    df = st.session_state.df
    try:
        value_num = pd.to_numeric(value, errors='ignore')
    except Exception:
        value_num = value
    if operator == 'greater_than':
        return df[df[column] > value_num]
    elif operator == 'less_than':
        return df[df[column] < value_num]
    elif operator == 'equals':
        return df[df[column] == value_num]
    else:
        return df

def sort_dataframe(column: str, ascending: bool = True):
    df = st.session_state.df
    return df.sort_values(by=column, ascending=ascending)

def group_and_aggregate(group_by_col: str, agg_col: str, agg_func: str):
    df = st.session_state.df
    if agg_func == 'sum':
        return df.groupby(group_by_col)[agg_col].sum().reset_index()
    elif agg_func == 'mean':
        return df.groupby(group_by_col)[agg_col].mean().reset_index()
    elif agg_func == 'count':
        return df.groupby(group_by_col)[agg_col].count().reset_index()
    elif agg_func == 'max':
        return df.groupby(group_by_col)[agg_col].max().reset_index()
    elif agg_func == 'min':
        return df.groupby(group_by_col)[agg_col].min().reset_index()
    else:
        return df

def pivot_table(index_col: str, columns_col: str, values_col: str, agg_func: str = 'sum'):
    df = st.session_state.df
    try:
        pivot_df = df.pivot_table(
            index=index_col, 
            columns=columns_col, 
            values=values_col, 
            aggfunc=agg_func,
            fill_value=0
        ).reset_index()
        return pivot_df
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        return df

def create_visualization(chart_type: str, x_col: str, y_col: str, title: str = None):
    df = st.session_state.df
    if df.empty:
        return None
    
    if chart_type == 'bar':
        return px.bar(df, x=x_col, y=y_col, title=title or f"Bar Chart: {y_col} by {x_col}")
    elif chart_type == 'line':
        return px.line(df, x=x_col, y=y_col, title=title or f"Line Chart: {y_col} by {x_col}")
    elif chart_type == 'pie':
        return px.pie(df, names=x_col, values=y_col, title=title or f"Pie Chart: {y_col} by {x_col}")
    elif chart_type == 'scatter':
        return px.scatter(df, x=x_col, y=y_col, title=title or f"Scatter Plot: {y_col} vs {x_col}")
    elif chart_type == 'histogram':
        return px.histogram(df, x=x_col, title=title or f"Histogram: {x_col}")
    elif chart_type == 'box':
        return px.box(df, x=x_col, y=y_col, title=title or f"Box Plot: {y_col} by {x_col}")
    else:
        return px.bar(df, x=x_col, y=y_col, title=title or f"Chart: {y_col} by {x_col}")


# Function declarations will be created when needed for Gemini API

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'original_df' not in st.session_state:
    st.session_state.original_df = pd.DataFrame()
if 'operation_history' not in st.session_state:
    st.session_state.operation_history = []
if 'last_command' not in st.session_state:
    st.session_state.last_command = ""
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'table'
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'selected_suggestion' not in st.session_state:
    st.session_state.selected_suggestion = None


def call_groq_api(prompt, max_tokens=2000):
    """Call Groq API for text generation"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=data)
        
        # Better error handling
        if response.status_code != 200:
            error_detail = response.text
            st.error(f"Groq API error {response.status_code}: {error_detail}")
            return None
            
        response_data = response.json()
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            st.error("No response from Groq API")
            return None
            
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"Network error calling Groq API: {e}")
        return None
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

def call_gemini_api(prompt):
    """Call Gemini API (fallback)"""
    try:
        from google import genai
        from google.genai import types
        
        # Create function declarations for Gemini
        filter_decl = types.FunctionDeclaration(
            name="filter_dataframe",
            description="Filter dataframe by column, operator, and value",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": ["greater_than", "less_than", "equals"]},
                    "value": {"type": "string"},
                },
                "required": ["column", "operator", "value"],
            },
        )

        sort_decl = types.FunctionDeclaration(
            name="sort_dataframe",
            description="Sort dataframe by column and order",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "ascending": {"type": "boolean"},
                },
                "required": ["column"],
            },
        )

        group_decl = types.FunctionDeclaration(
            name="group_and_aggregate",
            description="Group dataframe and apply aggregation",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "group_by_col": {"type": "string"},
                    "agg_col": {"type": "string"},
                    "agg_func": {"type": "string", "enum": ["sum", "mean", "count", "max", "min"]},
                },
                "required": ["group_by_col", "agg_col", "agg_func"],
            },
        )

        pivot_decl = types.FunctionDeclaration(
            name="pivot_table",
            description="Create a pivot table from the dataframe",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "index_col": {"type": "string"},
                    "columns_col": {"type": "string"},
                    "values_col": {"type": "string"},
                    "agg_func": {"type": "string", "enum": ["sum", "mean", "count", "max", "min"], "default": "sum"},
                },
                "required": ["index_col", "columns_col", "values_col"],
            },
        )

        visualization_decl = types.FunctionDeclaration(
            name="create_visualization",
            description="Create a chart visualization of the data",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "histogram", "box"]},
                    "x_col": {"type": "string"},
                    "y_col": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["chart_type", "x_col", "y_col"],
            },
        )

        tools = [
            types.Tool(function_declarations=[filter_decl]),
            types.Tool(function_declarations=[sort_decl]),
            types.Tool(function_declarations=[group_decl]),
            types.Tool(function_declarations=[pivot_decl]),
            types.Tool(function_declarations=[visualization_decl]),
        ]
        
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig()
            )
        )
        return response
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

def generate_multiple_suggestions(command, df_info):
    try:
        df_columns = df_info.columns.tolist()
        df_description = df_info.describe().to_string()
        prompt = (
            f"Given the following data with columns: {df_columns}, and the summary statistics:\n"
            f"{df_description}\n"
            f"The user wants to: '{command}'.\n"
            f"Generate 3 different interpretations of this request. Each interpretation should:\n"
            f"1. Use a different approach or focus\n"
            f"2. Be specific about which columns to use\n"
            f"3. Include a chart type if visualization is appropriate\n"
            f"4. Be clear and actionable\n"
            f"Return as JSON array with format: [{{'description': 'what this does', 'function': 'function_name', 'args': {{...}}, 'chart_type': 'optional'}}]"
        )

        ai_client = get_ai_client()
        
        # Try external APIs first, fallback to local processing
        if ai_client == "groq":
            try:
                response_text = call_groq_api(prompt)
                if not response_text:
                    raise Exception("Groq API failed")
            except Exception as e:
                st.warning(f"Groq API failed: {e}, using local suggestions")
                return create_local_suggestions(command, df_info)
        elif ai_client == "gemini":
            try:
                response = call_gemini_api(prompt)
                if not response:
                    raise Exception("Gemini API failed")
                response_text = response.text
            except Exception as e:
                st.warning(f"Gemini API failed: {e}, using local suggestions")
                return create_local_suggestions(command, df_info)
        else:
            # Use local processing
            return create_local_suggestions(command, df_info)

        # Parse the response to extract suggestions
        try:
            # Extract JSON from the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                suggestions_text = response_text[json_start:json_end]
            elif "[" in response_text and "]" in response_text:
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                suggestions_text = response_text[json_start:json_end]
            else:
                suggestions_text = response_text
            
            suggestions = json.loads(suggestions_text.strip())
            return suggestions[:3]  # Return max 3 suggestions
        except json.JSONDecodeError:
            # Fallback: create basic suggestions
            return [
                {
                    "description": f"Show data filtered and sorted for: {command}",
                    "function": "filter_dataframe",
                    "args": {"column": df_columns[0], "operator": "equals", "value": "all"},
                    "chart_type": "bar"
                },
                {
                    "description": f"Group and aggregate data for: {command}",
                    "function": "group_and_aggregate", 
                    "args": {"group_by_col": df_columns[0], "agg_col": df_columns[1] if len(df_columns) > 1 else df_columns[0], "agg_func": "sum"},
                    "chart_type": "pie"
                },
                {
                    "description": f"Create visualization for: {command}",
                    "function": "create_visualization",
                    "args": {"chart_type": "bar", "x_col": df_columns[0], "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0]},
                    "chart_type": "bar"
                }
            ]
    except Exception as e:
        st.error(f"An error occurred while generating suggestions: {e}")
        return []

def process_command_with_llm(command, df_info):
    try:
        df_columns = df_info.columns.tolist()
        df_description = df_info.describe().to_string()
        
        # Create a simple prompt for direct execution
        prompt = (
            f"Given the following data with columns: {df_columns}, and the summary statistics:\n"
            f"{df_description}\n"
            f"The user wants to: '{command}'.\n\n"
            f"IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:\n"
            f"{{\n"
            f'  "function": "function_name",\n'
            f'  "args": {{...}}\n'
            f"}}\n\n"
            f"Available functions and their parameters:\n"
            f"- filter_dataframe: {{'column': 'col_name', 'operator': 'greater_than/less_than/equals', 'value': 'value'}}\n"
            f"- sort_dataframe: {{'column': 'col_name', 'ascending': true/false}}\n"
            f"- group_and_aggregate: {{'group_by_col': 'col_name', 'agg_col': 'col_name', 'agg_func': 'sum/mean/count/max/min'}}\n"
            f"- pivot_table: {{'index_col': 'col_name', 'columns_col': 'col_name', 'values_col': 'col_name', 'agg_func': 'sum/mean/count/max/min'}}\n"
            f"- create_visualization: {{'chart_type': 'bar/line/pie/scatter/histogram/box', 'x_col': 'col_name', 'y_col': 'col_name', 'title': 'optional'}}\n\n"
            f"Choose the most appropriate function and provide the exact JSON response:"
        )

        ai_client = get_ai_client()
        
        # Try external APIs first, fallback to local processing
        if ai_client == "groq":
            try:
                response_text = call_groq_api(prompt)
                if response_text:
                    # Parse JSON response
                    try:
                        # Clean up the response text
                        if "```json" in response_text:
                            json_start = response_text.find("```json") + 7
                            json_end = response_text.find("```", json_start)
                            response_text = response_text[json_start:json_end]
                        elif "{" in response_text and "}" in response_text:
                            json_start = response_text.find("{")
                            json_end = response_text.rfind("}") + 1
                            response_text = response_text[json_start:json_end]
                        
                        # Try to parse the JSON
                        result = json.loads(response_text.strip())
                        
                        # Check if we have the required keys
                        if 'function' not in result or 'args' not in result:
                            st.warning(f"AI response missing required keys. Got: {list(result.keys())}")
                            return local_ai_processor(command, df_info)
                        
                        # Convert to function call format
                        class FunctionCall:
                            def __init__(self, name, args):
                                self.name = name
                                self.args = args
                        
                        return FunctionCall(result['function'], result['args'])
                    except (json.JSONDecodeError, KeyError) as e:
                        st.warning(f"Could not parse AI response: {e}")
                        return local_ai_processor(command, df_info)
                else:
                    st.warning("Groq API failed, using local processing")
                    return local_ai_processor(command, df_info)
            except Exception as e:
                st.warning(f"Groq API error: {e}, using local processing")
                return local_ai_processor(command, df_info)
                
        elif ai_client == "gemini":
            try:
                response = call_gemini_api(prompt)
                if response:
                    function_calls = response.function_calls
                    if function_calls and len(function_calls) > 0:
                        return function_calls[0]
                    else:
                        return local_ai_processor(command, df_info)
                else:
                    st.warning("Gemini API failed, using local processing")
                    return local_ai_processor(command, df_info)
            except Exception as e:
                st.warning(f"Gemini API error: {e}, using local processing")
                return local_ai_processor(command, df_info)
        else:
            # Use local processing
            st.info("Using local processing (no API keys configured)")
            return local_ai_processor(command, df_info)
    except Exception as e:
        st.error(f"An error occurred while processing your request: {e}")
        st.info("The model may not have been able to find a suitable tool.")
        return None

def create_local_suggestions(command, df_info):
    """Create local suggestions without external API calls"""
    df_columns = df_info.columns.tolist()
    command_lower = command.lower()
    
    suggestions = []
    
    # Suggestion 1: Visualization
    if any(word in command_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
        chart_type = "bar"
        if 'pie' in command_lower:
            chart_type = "pie"
        elif 'line' in command_lower:
            chart_type = "line"
        elif 'scatter' in command_lower:
            chart_type = "scatter"
            
        suggestions.append({
            "description": f"Create a {chart_type} chart visualization",
            "function": "create_visualization",
            "args": {
                "chart_type": chart_type,
                "x_col": df_columns[0],
                "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0]
            },
            "chart_type": chart_type
        })
    
    # Suggestion 2: Group and Aggregate
    if any(word in command_lower for word in ['group', 'sum', 'total', 'aggregate', 'by']):
        agg_func = "sum"
        if 'average' in command_lower or 'mean' in command_lower:
            agg_func = "mean"
        elif 'count' in command_lower:
            agg_func = "count"
            
        suggestions.append({
            "description": f"Group data and calculate {agg_func}",
            "function": "group_and_aggregate",
            "args": {
                "group_by_col": df_columns[0],
                "agg_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
                "agg_func": agg_func
            },
            "chart_type": "bar"
        })
    
    # Suggestion 3: Sort
    if any(word in command_lower for word in ['sort', 'order', 'top', 'bottom']):
        ascending = "top" not in command_lower
        suggestions.append({
            "description": f"Sort data by {df_columns[0]} ({'ascending' if ascending else 'descending'})",
            "function": "sort_dataframe",
            "args": {
                "column": df_columns[0],
                "ascending": ascending
            },
            "chart_type": "bar"
        })
    
    # If no specific suggestions, create generic ones
    if not suggestions:
        suggestions = [
            {
                "description": f"Create a bar chart of {df_columns[0]} vs {df_columns[1] if len(df_columns) > 1 else df_columns[0]}",
                "function": "create_visualization",
                "args": {
                    "chart_type": "bar",
                    "x_col": df_columns[0],
                    "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0]
                },
                "chart_type": "bar"
            },
            {
                "description": f"Group by {df_columns[0]} and sum {df_columns[1] if len(df_columns) > 1 else df_columns[0]}",
                "function": "group_and_aggregate",
                "args": {
                    "group_by_col": df_columns[0],
                    "agg_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
                    "agg_func": "sum"
                },
                "chart_type": "pie"
            },
            {
                "description": f"Sort data by {df_columns[0]}",
                "function": "sort_dataframe",
                "args": {
                    "column": df_columns[0],
                    "ascending": False
                },
                "chart_type": "bar"
            }
        ]
    
    return suggestions[:3]  # Return max 3 suggestions

def local_ai_processor(command, df_info):
    """Local AI processor using keyword matching - no external API needed"""
    df_columns = df_info.columns.tolist()
    command_lower = command.lower()
    
    class FunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args
    
    # Enhanced keyword matching
    if any(word in command_lower for word in ['chart', 'graph', 'plot', 'visualize', 'pie', 'bar', 'line', 'scatter']):
        chart_type = "bar"
        if 'pie' in command_lower:
            chart_type = "pie"
        elif 'line' in command_lower:
            chart_type = "line"
        elif 'scatter' in command_lower:
            chart_type = "scatter"
        elif 'histogram' in command_lower:
            chart_type = "histogram"
        elif 'box' in command_lower:
            chart_type = "box"
            
        return FunctionCall("create_visualization", {
            "chart_type": chart_type,
            "x_col": df_columns[0],
            "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "title": f"Chart: {command}"
        })
    
    elif any(word in command_lower for word in ['group', 'sum', 'total', 'aggregate', 'by']):
        agg_func = "sum"
        if 'average' in command_lower or 'mean' in command_lower:
            agg_func = "mean"
        elif 'count' in command_lower:
            agg_func = "count"
        elif 'max' in command_lower or 'maximum' in command_lower:
            agg_func = "max"
        elif 'min' in command_lower or 'minimum' in command_lower:
            agg_func = "min"
            
        return FunctionCall("group_and_aggregate", {
            "group_by_col": df_columns[0],
            "agg_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "agg_func": agg_func
        })
    
    elif any(word in command_lower for word in ['sort', 'order', 'top', 'bottom', 'highest', 'lowest']):
        ascending = True
        if any(word in command_lower for word in ['top', 'highest', 'desc']):
            ascending = False
            
        return FunctionCall("sort_dataframe", {
            "column": df_columns[0],
            "ascending": ascending
        })
    
    elif any(word in command_lower for word in ['pivot', 'cross', 'table']):
        return FunctionCall("pivot_table", {
            "index_col": df_columns[0],
            "columns_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "values_col": df_columns[2] if len(df_columns) > 2 else df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "agg_func": "sum"
        })
    
    elif any(word in command_lower for word in ['filter', 'where', 'show only', 'find']):
        return FunctionCall("filter_dataframe", {
            "column": df_columns[0],
            "operator": "equals",
            "value": "all"
        })
    
    else:
        # Default to showing the data with a chart
        return FunctionCall("create_visualization", {
            "chart_type": "bar",
            "x_col": df_columns[0],
            "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "title": f"Data View: {command}"
        })

def create_fallback_suggestion(command, df_columns):
    """Create a simple fallback suggestion when AI response is malformed"""
    class FunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args
    
    # Simple heuristics based on command keywords
    command_lower = command.lower()
    
    if any(word in command_lower for word in ['chart', 'graph', 'plot', 'visualize', 'pie', 'bar', 'line']):
        return FunctionCall("create_visualization", {
            "chart_type": "bar",
            "x_col": df_columns[0],
            "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0]
        })
    elif any(word in command_lower for word in ['group', 'sum', 'total', 'aggregate']):
        return FunctionCall("group_and_aggregate", {
            "group_by_col": df_columns[0],
            "agg_col": df_columns[1] if len(df_columns) > 1 else df_columns[0],
            "agg_func": "sum"
        })
    elif any(word in command_lower for word in ['sort', 'order', 'top', 'bottom']):
        return FunctionCall("sort_dataframe", {
            "column": df_columns[0],
            "ascending": "top" not in command_lower
        })
    else:
        # Default to showing the data
        return FunctionCall("create_visualization", {
            "chart_type": "bar",
            "x_col": df_columns[0],
            "y_col": df_columns[1] if len(df_columns) > 1 else df_columns[0]
        })

def apply_suggestion(function_name, function_args, description):
    """Apply a suggestion to the dataframe"""
    try:
        if function_name in functions:
            new_df = functions[function_name](**function_args)
            if isinstance(new_df, pd.DataFrame):
                st.session_state.df = new_df
                st.session_state.operation_history.append({
                    'description': description,
                    'params': function_args,
                    'function': function_name
                })
                st.success(f"Applied: {description}")
                st.session_state.current_view = 'table'
            else:
                st.error("Function did not return a DataFrame.")
        else:
            st.error(f"Unknown function: {function_name}")
    except Exception as e:
        st.error(f"Failed to apply the function: {e}")

# Map the function names to Python functions for local execution
functions = {
    "filter_dataframe": filter_dataframe,
    "sort_dataframe": sort_dataframe,
    "group_and_aggregate": group_and_aggregate,
    "pivot_table": pivot_table,
    "create_visualization": create_visualization,
}

# Streamlit app UI and logic --------------------------------------------------

st.set_page_config(layout="wide", page_title="NL Data Explorer", page_icon="📈")
st.title("📊 NL Data Explorer")
st.markdown("Talk, don't tool. Describe your needs in plain language and get a useful view.")

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

if not st.session_state.df.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("What do you want to see?")
        command = st.text_input("Enter your command here...", value=st.session_state.last_command, key="command_input")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Get Suggestions", type="primary"):
                if command:
                    st.session_state.last_command = command
                    with st.spinner("Generating suggestions..."):
                        st.session_state.suggestions = generate_multiple_suggestions(command, st.session_state.original_df)
                else:
                    st.warning("Please enter a command.")
        
        with col_btn2:
            if st.button("Apply Directly"):
                if command:
                    st.session_state.last_command = command
                    with st.spinner("Thinking..."):
                        llm_call = process_command_with_llm(command, st.session_state.original_df)

                    if llm_call:
                        function_name = llm_call.name
                        function_args = llm_call.args
                        apply_suggestion(function_name, function_args, "Direct application")
                    else:
                        st.warning("The model could not determine a suitable action. Try getting suggestions instead.")
                else:
                    st.warning("Please enter a command.")

        # Suggestions Panel
        if st.session_state.suggestions:
            st.markdown("---")
            st.subheader("💡 Choose Your Interpretation")
            st.markdown("**Your command:** " + command)
            st.markdown("**Here are 3 different ways to interpret your request:**")
            
            for i, suggestion in enumerate(st.session_state.suggestions):
                with st.expander(f"Option {i+1}: {suggestion['description']}", expanded=False):
                    st.json(suggestion)
                    if st.button(f"Apply Option {i+1}", key=f"apply_{i}"):
                        apply_suggestion(suggestion['function'], suggestion['args'], suggestion['description'])
                        st.session_state.suggestions = []  # Clear suggestions after applying
                        st.rerun()
    with col2:
        st.subheader("Current View Explained")
        if st.session_state.operation_history:
            last_op = st.session_state.operation_history[-1]
            st.markdown(f"**You are currently viewing data based on this command:**")
            st.markdown(f"**Description:** `{last_op['description']}`")
            st.json(last_op['params'])
        else:
            st.info("No operations applied yet. This is a preview of the raw data.")

    st.markdown("---")
    st.subheader("Data View")

    if st.session_state.current_view == 'table':
        st.dataframe(st.session_state.df)

        if len(st.session_state.df.columns) >= 2:
            st.markdown("---")
            st.subheader("📊 Quick Chart Visualizations")
            
            # Chart type selector
            chart_types = ['bar', 'line', 'pie', 'scatter', 'histogram', 'box']
            selected_charts = st.multiselect(
                "Select chart types to display:",
                chart_types,
                default=['bar', 'pie']
            )
            
            x_col = st.session_state.df.columns[0]
            y_col = st.session_state.df.columns[1] if len(st.session_state.df.columns) > 1 else st.session_state.df.columns[0]
            
            # Display selected charts
            for chart_type in selected_charts:
                try:
                    fig = create_visualization(chart_type, x_col, y_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create {chart_type} chart: {e}")

            st.info("💡 **Tip:** Ask for specific charts using commands like 'show a pie chart of sales by region' or 'create a bar graph of top products'.")

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
