# JupyterLLMAgents
A collection of Jupyter notebooks that explore utilizing LLM (large language model) agents for interactive analysis.

### Requirements
This project assumes the use of python 3.9 but may work with other python versions as well. A [OpenAI API key](https://platform.openai.com/account/api-keys) is required and should be set in an environment variable named `OPENAI_API_KEY`.

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
