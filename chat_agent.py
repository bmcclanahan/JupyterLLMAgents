from IPython.core.magic import magics_class, register_line_cell_magic, Magics

import ast
import astor

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


from typing import Any, Optional, Dict, List

from constants import WHITELISTED_LIBRARIES, WHITELISTED_BUILTINS


class ChatAgentCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.descriptions = []

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        index = action.log.find("Action:")
        if index != -1:
            self.descriptions.append(
                 action.log[:index].strip()
            )
            print('description extracted', self.descriptions[-1])
                  
    

@magics_class
class ChatAgentMagics(Magics):

    def __init__(self):
        #super(ChatAgentMagics, self).__init__(shell)  uncomment this when making this a proper jupyter extension loaded with %load_ext
        self.__agent_input = {}
        self.__llm = OpenAI(temperature=0)
        self.__tools = [
            Tool.from_function(
                func=self.perform_dataframe_analysis,
                name = 'dataframeAnalysis',
                description='Tool to perform analysis on Pandas dataframes. Input should be python code containing dataframe operations to derive answers to questions. The input code should store the answer in variable named result.'
            )
        ]
        self.__agent  = initialize_agent(
            self.__tools, self.__llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3
        )
        self.__callback_handler = ChatAgentCallbackHandler()

    def is_df_overwrite(self, node: ast.stmt) -> str:

        """
        Remove df declarations from the code to prevent malicious code execution. A helper method.
        Args:
            node (object): ast.stmt

        Returns (str):

        """

        return (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and re.match(r"df\d{0,2}$", node.targets[0].id)
        )

    def is_unsafe_import(self, node: ast.stmt) -> bool:

        """Remove non-whitelisted imports from the code to prevent malicious code execution

        Args:
            node (object): ast.stmt

        Returns (bool): A flag if unsafe_imports found.

        """

        return isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name not in WHITELISTED_LIBRARIES for alias in node.names
        )

    def clean_code(self, code: str) -> str:

        """
        A method to clean the code to prevent malicious code execution
        Args:
            code(str): A python code

        Returns (str): Returns a Clean Code String

        """
        tree = ast.parse(code)

        new_body = [
            node
            for node in tree.body
            if not (self.is_unsafe_import(node) or self.is_df_overwrite(node))
        ]

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree).strip()
    
    def set_agent_input(self, agent_input: dict):
        self.__agent_input = agent_input

    def get_agent_input(self):
        return self.__agent_input

    def perform_dataframe_analysis(self, analysis_code: str):
        analysis_code = self.clean_code(analysis_code)
        print()
        print('cleaned code')
        print(analysis_code)
        input_environment = {
            key: self.__agent_input[key]['value']
            for key in self.__agent_input
        }
        environment = {
            **input_environment,
            "__builtins__": {
                **{
                    builtin: __builtins__[builtin]
                    for builtin in WHITELISTED_BUILTINS
                },
            },
        }
        exec(analysis_code, environment)
        last_line = analysis_code.split('\n')[-1].strip()
        if 'result' in last_line and 'result' in environment:
            result = environment['result']
        else:
            result = eval(last_line, environment)
        result_num = len([key for key in self.__agent_input if 'result' in key])
        result_key = f'result{result_num + 1}'
        self.__agent_input[result_key] = {
            'value': result, 'description': self.__callback_handler.descriptions[-1]
        }
        return f'Answer has been successfully derived. Key: {result_key}' if not type(result) == str else result

    def chat_agent(self, line: Optional[str], cell: Optional[str]=None):
        "Magic that works both as %lcmagic and as %%lcmagic"
        if cell is None:
            response = agent.run(line)
        else:
            available_variables = '\n'.join(
                [
                    key + ' - ' + self.__agent_input[key]['description']
                    for key in self.__agent_input
                ]
            )
            cell = cell + """\nWhen using the dataframeAnalysis tool you may assume that you have access to the following variables when writing the code:
            """ + available_variables
            print('prompt is ')
            print(cell)
            response = self.__agent.run(cell, callbacks=[self.__callback_handler])
        return response


chat_agent_magic = ChatAgentMagics()

set_inputs = chat_agent_magic.set_agent_input
get_inputs = chat_agent_magic.get_agent_input

def get_result(key: str):
    return get_inputs()[key]['value']

register_line_cell_magic(chat_agent_magic.chat_agent)

# def load_ipython_extension(ipython):
#     ipython.register_magics(chat_agent_magic)