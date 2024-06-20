# Project Name

Welcome to the Japanese Search RAG using LLM. This is an combination of RAG and LLM Agent. User can upload a csv file. After that according to the user's query the LLM will either retrieve data from the csv file or search answers from the web

## Table of Contents

- Introduction
- Installation
- Hyperparameters
- Inference
- Demo

## Introduction

Search Rag is an LLM Application. After running the application you can upload a CSV file. This CSV file may contain any dataset or any other information. After uploading the CSV file, user can ask any questions. The model generates an output depending on the question as follows,

- If the query is related to the uploaded dataset (CSV file) then the LLM will use RAG to extract information from the dataset and provide accurate answer of that query.
- If the query is not related to the uploaded dataset (CSV file) then the LLM will use API to get result from a search engine (SERP API) and provide answer.

## Installation

To get started, you need to set up the Conda environment.

### Step 1: Install Conda

If you haven't already, install Conda from the [official Anaconda website](https://www.anaconda.com/products/distribution) and follow the installation instructions.

### Step 2: Create the Conda environment

Once Conda is installed, create a new environment named `llm_module` using the provided `.yml` file and activate that environment:

```bash
conda env create -f environment.yml
conda activate langchain2
```

## Hyperparameters

Before running the app, you ned to set SERP API KEY. You can also change the base model and embed model. To do these changes you have to edit the 'config.py' file

```python
SERP_API = 'INSERT_YOUR_SERP_API_KEY_HERE'

embed_model = "intfloat/multilingual-e5-large"
base_model = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
```

## Inference
To run the app you just hape to run the following command,
```bash
python app.py
```

## Demo

  <video width="800" height="360" controls>
    <source src="Demo/NSL_SEARCH_RAG.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

