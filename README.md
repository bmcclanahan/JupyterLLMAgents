# JupyterLLMAgents
A collection of Jupyter notebooks that explore utilizing LLM (large language model) agents for interactive analysis.
This project includes a chat agent tool which can be accessed as a Jupyter Magic command `%%chat_agent`. By placing the `%%chat_agent` command at the top of a cell you can instruct a LLM to execute different actions and store the result in your notebook environment (e.g. performing analysis on a Pandas Dataframe). Here's an example.
```
%%chat_agent -n
Retrieve the price of Bitcoin and store it in a variable named btc_pr
```
By executing a cell that uses the `chat_agent` magic command you can retrieve the current price of Bitcoin and have it stored in a variable named `btc_pr`. The `get_result` function from the `chat_agent` module can be used to retrieve variables produced by the chat agent.

```
chat_agent.get_result('btc_pr')
```
These variables can then be accessed by the chat agent for subsequent references and computations.

Reference the **HelloAgent** notebook for a complete example.

### Requirements
This project assumes the use of python 3.9 but may work with other python versions as well. A [OpenAI API key](https://platform.openai.com/account/api-keys) is required and should be set in an environment variable named `OPENAI_API_KEY`.

The **HelloAgent** notebook also requires you to have an [API key](https://serper.dev/) for the Google Search API. The `SERPER_API_KEY` environment variable should be set with this key. 

### Installation
Dependencies are managed with [pipenv](https://pipenv.pypa.io/en/latest/). You can install pipenv by running
```
pip install -r requirements.txt
```
from this project's root directory.

To install all required dependencies to run the Jupyter notebooks included in this project run.
```
pipenv install
```

### Running Notebooks

After installation with `pipenv` run
```
pipenv run jupyter lab
```
to start a jupyter lab server in an environment where all notebook dependencies are included. By default you can connect to the server on `localhost:8888`.
