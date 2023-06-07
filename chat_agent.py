from IPython.core.magic import magics_class, register_line_cell_magic, Magics

import ast
import argparse
import astor
import pandas as pd
import re
import os

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.utilities import GoogleSerperAPIWrapper


from typing import Any, Optional, Dict, List

from constants import WHITELISTED_LIBRARIES, WHITELISTED_BUILTINS


class ChatAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.descriptions = []
        self.agent_action = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print("on action")
        index = action.log.find("Action:")
        self.agent_action = action
        if index != -1:
            self.descriptions.append(action.log[:index].strip())


@magics_class
class ChatAgentMagics(Magics):
    def __init__(self):
        # super(ChatAgentMagics, self).__init__(shell)  uncomment this when making this a proper jupyter extension loaded with %load_ext
        self.__agent_input = {}
        self.__llm = OpenAI(temperature=0)
        tools = (
            load_tools(["google-serper"], llm=self.__llm)
            if "SERPER_API_KEY" in os.environ
            else []
        )
        self.__tools = tools + [
            Tool.from_function(
                func=self.python_execution,
                name="pythonCodeExecution",
                description="Tool used to execute Python code. Input should be python code containing statements to derive answers to questions or solutions to instructions. The input code should store the answer in variable named result unless instructed otherwise. The tool may return feedback from the user on the input code. If the result is a numeric value be sure to assign it to a variable with proper formatting without commas, dollar signs,  percent symbols or any other symbol.",
            )
        ]
        self.__agent = initialize_agent(
            self.__tools,
            self.__llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
        )
        self.__callback_handler = ChatAgentCallbackHandler()
        self.__noninteractive = False
        self.__verbose = False
        self.__last_key = None

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

    def add_agent_input(
        self,
        input_var: Any,
        name: str,
        description: str,
        rows: int = 5,
        include_df_head: bool = True,
    ):
        if type(input_var) == pd.DataFrame and include_df_head:
            description += f"""
This is the result of `print(df.head({rows}))`
{input_var.head(rows)}
        """
        self.__agent_input[name] = {"value": input_var, "description": description}

    def set_agent_input(self, agent_input: dict):
        self.__agent_input = agent_input

    def get_agent_input(self):
        return self.__agent_input

    def python_execution(self, analysis_code: str):
        last_character = self.__callback_handler.agent_action.log.strip()[-1]
        if last_character == '"' and not analysis_code.endswith('"'):
            analysis_code += '"'  # replace missing quotes that langchain strips
        try:
            analysis_code = self.clean_code(analysis_code)
            print()
            if self.__verbose:
                print("input code")
                print(analysis_code)
            user_feedback = ""
            if not self.__noninteractive:
                prompt = f"""
    
    The change agent would like to run the following code:
    
    --------------------------------------------------------
    {analysis_code}
    --------------------------------------------------------
    
    To allow execution type Y or type N to disallow.
    You may give additional feedback to the option for either option by placing a dash after the allow option followed by the feedbak. For example:
    Y - this code answers my original question
    or
    N - this code does not produce the right answer
    
                """
                feedback_retrieved = False
                while not feedback_retrieved:
                    try:
                        user_input = input(prompt)
                        user_input = user_input.strip().split("-")
                        first_input = user_input[0].strip().lower()
                        if first_input not in ("y", "n"):
                            raise ValueError("Must enter Y or N")
                        if len(user_input) > 1:
                            user_feedback = " - ".join(user_input[1:])
                        if first_input == "n":
                            response_end = (
                                "most likely because it doesn't achieve the desired result."
                                if len(user_feedback) == 0
                                else f" and has the following feedback: {user_feedback}"
                            )
                            return f"The user disallowed execution of the code{response_end}"
                        feedback_retrieved = True
                    except ValueError as e:
                        print(e)
                        pass

            input_environment = {
                key: self.__agent_input[key]["value"] for key in self.__agent_input
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

            code_parse = ast.parse(analysis_code, mode="exec")
            key_val = "result"
            if "result" in environment:
                result = environment["result"]
            elif type(code_parse.body[-1]) == ast.Assign:
                if self.__verbose:
                    print(
                        "The variable `result` was not found in executing environment. Using the assignment on the last code line instead for the result."
                    )
                key_val = code_parse.body[-1].targets[0].id
                result = environment[key_val]
            else:
                return "complete. No assignment operation found in last lines of code."
            result_num = len([key for key in self.__agent_input if key_val in key])
            result_key = key_val + (str(result_num + 1) if result_num > 0 else "")
            alias_description = f"It is an alias for {key_val} and" if key_val else ""
            description = f'object of type {type(result)} related to the thought "{self.__callback_handler.descriptions[-1]}"'
            if type(result) == pd.DataFrame:
                description += (
                    f". The dataframe has the columns {result.columns.values}"
                )
            print("saving result to agent input ", result_key)
            self.__agent_input[result_key] = {
                "value": result,
                "description": description,
            }
            response_end = (
                ""
                if len(user_feedback) == 0
                else f" - The user has the following feedback: {user_feedback}"
            )
            self.__last_key = result_key
            return (
                f"Answer has been successfully derived. Key: {result_key}{response_end}"
                if not type(result) == str
                else result + response_end
            )
        except Exception as e:
            return f"execution failed with the error message: {str(e)}"

    def chat_agent(self, line: Optional[str], cell: Optional[str] = None):
        "Magic that works as %%chat_agent"
        options = list(filter(lambda x: len(x) != 0, line.strip().split(" ")))
        parser = argparse.ArgumentParser(description="chat agent options")
        parser.add_argument(
            "--noninteractive",
            "-n",
            action="store_true",
            help="runs the agent in a non interactive mode where the user is not prompted for input",
            default=False,
            required=False,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="verbose option",
            default=False,
            required=False,
        )
        args = parser.parse_args(options)
        self.__noninteractive = args.noninteractive
        self.__verbose = args.verbose
        available_variables = "\n\n".join(
            [
                key + " - " + self.__agent_input[key]["description"]
                for key in self.__agent_input
            ]
        )
        cell = (
            cell
            + (
                """\nWhen using the pythonCodeExecution tool you may assume that you have access to the following variables when writing the code:

"""
                if len(self.__agent_input) > 0
                else ""
            )
            + available_variables
        )
        cell = cell.strip()
        print("Prompt:")
        print(cell)
        response = self.__agent.run(cell, callbacks=[self.__callback_handler])
        return response


chat_agent_magic = ChatAgentMagics()

set_inputs = chat_agent_magic.set_agent_input
get_inputs = chat_agent_magic.get_agent_input
add_agent_input = chat_agent_magic.add_agent_input


def get_result(key: str):
    return get_inputs()[key]["value"]


register_line_cell_magic(chat_agent_magic.chat_agent)

# def load_ipython_extension(ipython):
#     ipython.register_magics(chat_agent_magic)
