# 📊 NL Data Explorer

A simple Streamlit application that allows non-technical users to explore datasets using natural language commands. This project meets the key requirements of a "Data Explorer with Natural Commands" by offering a conversational interface, suggesting views, and providing clear explanations of applied operations.

### ✨ **Features**

  * **CSV Uploader**: Easily upload your own datasets.
  * **Natural Language Command Box**: Describe your data needs in plain English.
  * **Multiple Interpretations**: Get 2-3 different ways to interpret vague commands.
  * **Suggestions Panel**: Choose from multiple options with confirm/apply workflow.
  * **Operation Explanation Panel**: Understand exactly how your data is being transformed.
  * **Rich Visualizations**: Pie charts, bar graphs, line graphs, scatter plots, histograms, and box plots.
  * **Pivot Tables**: Create pivot tables for advanced data analysis.
  * **Data Export**: Download the current view as a CSV file.
  * **Operation History**: Track all applied operations with clear explanations.

-----

### 🛠️ **Installation**

To get the application up and running, follow these steps. It is highly recommended to use a virtual environment.

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/NiteeshPutla/dataexplorer.git
    cd dataexplorer
    ```

2.  **Set up a Virtual Environment**

    ```bash
    # Create the virtual environment
    python -m venv dataexplore

    # Activate the virtual environment
    # On Windows
    dataexplore\Scripts\activate
    # On macOS and Linux
    source dataexplore/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file contains all the necessary libraries:

    ```
    streamlit==1.36.0
    pandas==2.2.2
    plotly==5.15.0
    google-generativeai
    ```

-----

### **Add API Key**

Create `.streamlit/secrets.toml` and add your API key. You can use either Groq (recommended) or Gemini:

**Option 1: Groq (Recommended - More generous free tier)**
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

**Option 2: Gemini (Fallback)**
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

**Getting API Keys:**
- **Groq**: Sign up at [console.groq.com](https://console.groq.com) - Free tier includes 14,400 requests per day
- **Gemini**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 🚀 **Usage**

With your virtual environment activated, run the application from the command line:

```bash
python -m streamlit run main.py
```

Your browser will automatically open a new tab with the application.

### 📝 **How to Use**

1.  **Upload a CSV file** using the uploader in the sidebar.
2.  **Type a command** in the text box, like `"show sales by region"` or `"top 5 products this quarter"`.
3.  **Choose your approach:**
    - **Get Suggestions**: Get 2-3 different interpretations of your command
    - **Apply Directly**: Let the AI choose the best interpretation automatically
4.  **Select an option** from the suggestions panel (if using suggestions mode)
5.  **View the results** in the main area with interactive charts and data tables
6.  **See explanations** of what was done in the right-hand panel
7.  **Export your view** as a CSV file when ready

### 🎯 **Example Commands**

- `"show seasonality by region"` - Creates visualizations showing seasonal patterns
- `"top 5 products this quarter"` - Filters and sorts data to show top performers
- `"create a pie chart of sales by category"` - Generates a pie chart visualization
- `"pivot table with regions as rows and months as columns"` - Creates a pivot table
- `"bar graph of revenue by product"` - Displays data as a bar chart

-----

### 📄 **License**

This project is licensed under the MIT License.