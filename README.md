# Obsidian RAG

Obsidian RAG is a powerful command-line tool that allows you to chat with your Obsidian notes. It uses a Retrieval Augmented Generation (RAG) model to provide intelligent, context-aware answers to your questions, based on the content of your vaults.

## Prerequisites

Before you begin, make sure you have the following installed:

*   **Python 3.7+**: You can download it from [python.org](https://python.org).
*   **Ollama**: This is the engine that runs the language model. You can download it from [ollama.ai](https://ollama.ai).
*   **Git**: You'll need this to clone the repository. You can download it from [git-scm.com](https://git-scm.com/).

## Getting Started

Follow these simple steps to get up and running:

**1. Clone the Repository**

Open your terminal and run the following command to clone the repository to your local machine:

```bash
git clone https://github.com/jacksalillas/obsidian-rag.git
cd obsidian-rag
```

**2. Create a Virtual Environment**

It's a good practice to create a virtual environment to keep the project's dependencies separate from your system's Python installation.

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

Install all the necessary Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**4. Start Ollama and Download the Model**

Before you can run the application, you need to start the Ollama server and download the `mistral` language model.

First, start the Ollama server in a separate terminal window:

```bash
ollama serve
```

Then, in another terminal window, download the `mistral` model:

```bash
ollama pull mistral:latest
```

**5. Configure Your Vaults**

The `config.json` file is where you tell the application where to find your Obsidian vaults. You can add the full paths to your vaults in the `vaults` list.

```json
{
  "vaults": [
    "/path/to/your/first/vault",
    "/path/to/your/second/vault"
  ],
  "text_sources": [],
  "model_name": "mistral:latest",
  "embedding_model_name": "all-MiniLM-L6-v2",
  "obsidian_root": null,
  "mnemosyne_vault_guidance": []
}
```

**6. Run the Application**

You're all set! Now you can start the application by running the `start_obsidian` script:

```bash
./start_obsidian
```

The application will index your vaults and then you can start asking questions about your notes.

## How it Works

The application uses the following components:

*   **Ollama**: Runs the `mistral` language model.
*   **ChromaDB**: Stores the vector embeddings of your notes.
*   **Sentence-Transformers**: Creates the vector embeddings.
*   **LangChain**: Provides the framework for the RAG model.

When you ask a question, the application searches for the most relevant notes in your vaults and then uses the language model to generate an answer based on the content of those notes.

## Contributing

Contributions are welcome! If you have any ideas for how to improve the application, please open an issue or submit a pull request.
