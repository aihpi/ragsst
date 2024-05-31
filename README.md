
# RAGSST

## Retrieval Augmented Generation and Semantic-search Tool

A quick start tool to test and use as basis for various document-related use cases locally:

- Rag: Prompt an LLM that uses relevant context to answer your queries
- Semantic Retrieval: Retrieve relevant passages from documents showing sources and relevance
- Rag Chat: Interact with an LLM making use of retrieval and history
- LLM Chat: Simply chat and test a local LLM, without document context


The interface is divided into tabs for users to select and try the tool for the desired use case. 
The implementation is focused on simplicity and low-level implementation in order to depict the working principles and components, allowing developers and Python enthusiasts to modify and build upon.


### Installation

Download or clone the repository

On bash, you can run the following installation script:

```shell
$ bin/install.sh
```

---

**Alternatively, install it manually:**

#### Create and activate a virtual environment

```shell
$ python3 -m venv .myvenv
$ source .myvenv/bin/activate
```

#### Install dependencies

```shell
$ pip3 install -r requirements.txt
```

#### Ollama

Install it to run large language models locally

```shell
$ curl -fsSL https://ollama.ai/install.sh | sh
```

Or follow the installation instructions for your operating system:

[Install Ollama](https://ollama.com/download)

choose a model and download it. For example

```shell
$ ollama pull llama3
```

### Usage

- Choose the desire settings on parameters.py
- Start it with 

```shell
$ python3 ragsst.py
```

- Open the expose url on your favourite browser
- Enjoy

## Development

Before commiting, format the code by using the following:

Formatter: black

On the project folder:

```shell
$ black -t py311 -S -l 99 .
```

Linters:

- Pylance
- flake8 (args: --max-line-length=100 --extend-ignore=E401,E501,E741)

---

#### Additional Input parameters for the LLMs

- Top k: Ranks the output tokens in descending order of probability, selects the first k tokens to create a new distribution, and it samples the output from it. Higher values result in more diverse answers, and lower values will produce more conservative answers.

- Top p: Works together with Top k, but instead of selecting a fixed number of tokens, it selects enough tokens to cover the given cumulative probability. A higher value will produce more varied text, and a lower value will lead to more focused and conservative answers.

- Temp: This affects the “randomness” of the answers  by scaling the probability distribution of the output elements. Increasing the temperature will make the model answer more creatively.

---

## License

[GPLv3](./LICENCE)
