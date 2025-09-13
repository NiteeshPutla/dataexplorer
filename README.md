# 📊 NL Data Explorer

A simple Streamlit application that allows non-technical users to explore datasets using natural language commands. This project meets the key requirements of a "Data Explorer with Natural Commands" by offering a conversational interface, suggesting views, and providing clear explanations of applied operations.

### ✨ **Features**

  * **CSV Uploader**: Easily upload your own datasets.
  * **Natural Language Command Box**: Describe your data needs in plain English.
  * **Intelligent Suggestions**: Get 2-3 interpretations of your command when it's vague.
  * **Operation Explanation Panel**: Understand exactly how your data is being transformed.
  * **Interactive Charting**: Visualize your data with a single click.
  * **Data Export**: Download the current view as a CSV file.

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
    ```

-----

### 🚀 **Usage**

With your virtual environment activated, run the application from the command line:

```bash
python -m streamlit run main.py
```

Your browser will automatically open a new tab with the application.

### 📝 **How to Use**

1.  **Upload a CSV file** using the uploader in the sidebar.
2.  **Type a command** in the text box, like `"show sales by region"` or `"top 5 products this quarter"`.
3.  **Click "Apply Command"** and select the interpretation that best matches your intent.
4.  **View the results** in the main area and see a summary of the operation in the right-hand panel.

-----

### 📄 **License**

This project is licensed under the MIT License.