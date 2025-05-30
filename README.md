# REM-Challenge-Task

## Description

This task demonstrates a workflow for processing audio from YouTube videos, converting the audio to text, building a vector database from the transcriptions, and utilizing a LangGraph-based RAG (Retrieval-Augmented Generation) system to answer questions and analyze the speaker's accent based on the transcribed content.

## Project Structure

```
REMWaste_task/
├── audio_to_text.py
├── chains.py
├── modules.ipynb
├── nodes.py
├── prompt/
│   └── retrieval_chain_prompt.yaml
├── requirements.txt
├── retrieval_chain.py
├── state.py
└── vectordb.py
```
## Key Components

-   `audio_to_text.py`: Contains the function to extract audio from a given YouTube URL and convert it into text using OpenAI Whisper.
-   `vectordb.py`: Includes functions for creating and managing a FAISS vector database from text data, utilizing cached embeddings for efficiency.
-   `state.py`: Defines the `GraphState` TypedDict used to manage the state within the LangGraph workflow.
-   `chains.py`: Houses the LangChain expression language chains for language checking, accent classification, and accent grading using OpenAI models with structured output.
-   `nodes.py`: Defines the nodes used in the LangGraph, wrapping the functionality from the chains and vector database retrieval.
-   `retrieval_chain.py`: Creates the final RAG chain that combines retrieved documents with accent analysis results to generate an answer.
-   `modules.ipynb`: A Jupyter notebook demonstrating the end-to-end workflow, including audio processing, vector database setup, LangGraph construction, and execution.
-   `prompt/retrieval_chain_prompt.yaml`: Directory containing prompt templates (e.g., in YAML format) used by the chains.
-   `requirements.txt`: Lists the Python dependencies required to run the project.

## Workflow Diagram

![output](https://github.com/user-attachments/assets/2cbcd952-0ed7-4ff2-8c02-f8df0c68c217)

## Usage

Open and run the `modules.ipynb` Jupyter notebook. The notebook guides you through the steps of processing a YouTube video, building the vector database, setting up the LangGraph workflow, and running queries.

## Workflow (as shown in `modules.ipynb`)

1.  Extract audio from a specified YouTube URL and transcribe it to text.
2.  Split the transcribed text into chunks and build a FAISS vector database.
3.  Initialize the LangGraph with defined nodes:
    -   RetrievalNode: Retrieves relevant text chunks based on the user question.
    -   LanguageCheckerNode: Checks if the language of the retrieved documents is English.
    -   AccentClassifierNode: Classifies the English accent type.
    -   AccentGraderNode: Grades the confidence level of the English accent.
    -   RetrievalAnswerNode: Generates a final answer using the retrieved documents and accent analysis results.
4.  Define the edges and conditional logic between the nodes.
5.  Compile and execute the LangGraph with a user question. 

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd REMWaste_task
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        .venv\Scripts\activate
        ```

    -   On macOS/Linux:

        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**

    Create a `.env` file in the `REMWaste_task` directory and add your OpenAI API key:

    ```dotenv
    OPENAI_API_KEY='your-openai-api-key'
    ```
