##########################################################################################
# following the instructions provided here:
# https://medium.com/@cubode/comprehensive-guide-using-ai-agents-to-analyze-and-process-csv-data-a0259e2af761
##########################################################################################

import json

import pandas as pd

from pathlib import Path
from typing import Dict, Annotated

from langchain_ollama import ChatOllama
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

# ---------
# Globals
# ---------

# Create an instance of PythonREPL
repl = PythonREPL()


def main():
    """MAIN"""

    # read the CSV and define it's meta-data
    csv_path = Path(Path(__file__).parent.parent, "data", "winemag-data-130k-v2.csv")
    df = pd.read_csv(csv_path)
    metadata = extract_metadata(df)
    print(json.dumps(metadata, indent=2))

    llm = ChatOllama(
        model="llama3",
        temperature=0.8,
        num_predict=256,
        # other params ...
    )

    prompt_template = """
    Assistant is an AI model that takes in metadata from a dataset 
    and suggests charts to use to visualise that data.

    New Input: Suggest 2 charts to visualise data from a dataset with the following metadata. 


    SCHEMA:

    -------- 

    {schema}

    DATA TYPES: 

    -------- 

    {data_types}

    SAMPLE: 

    -------- 

    {sample}

    """.format(
        schema=metadata["Schema"], data_types=metadata["Data Types"], sample=metadata["Sample"]
    )

    print("invoking LLM w/ our prompt template")
    x = llm.invoke(prompt_template)
    print(x)

    # Pass the dataframe into the globals dictionary of the PythonREPL instance
    repl.globals["df"] = df

    tools = [python_repl]

    instructions_template = """

    You are an agent that writes and excutes python code

    You have access to a Python abstract REPL, which you can use to execute the python code.

    You must write the python code assuming that the dataframe (stored as df) has already been read.

    If you get an error, debug your code and try again.

    You might know the answer without running any code, but you should still run the code to get the answer.

    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.

    Do not create example dataframes 
    """

    base_template = """

        {instructions_template}

        TOOLS:

        ------

        You have access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```

        Thought: Do I need to use a tool? Yes

        Action: the action to take, should be one of [{tool_names}]

        Action Input: the input to the action

        Observation: the result of the action

        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```

        Thought: Do I need to use a tool? No

        Final Answer: [your response here]

        ```

        Begin!

        Previous conversation history:

        {chat_history}

        New input: {input}

        {agent_scratchpad}
    """

    base_prompt = PromptTemplate(
        template=base_template,
        input_variables=["agent_scratchpad", "input", "instructions", "tool_names", "tools"],
    )
    base_prompt = base_prompt.partial(
        instructions_template=instructions_template
    )  # format the instructions

    new_input = """
    Create the data needed for the following chart

    CHART 

    -------- 
    Chart 1: Bar Chart - Top 10 Wineries by Average Points

    X-axis: Winery names

    Y-axis: Average points\n*

    You're task is to write code to transform the raw data into the desired data structure in json to be included in the plot.

    Only return the data.

    Here is the dataset schema:

    {schema}

    """.format(
        schema=metadata["Schema"]
    )

    agent = create_react_agent(llm, tools, base_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_out = agent_executor.invoke({"input": new_input, "chat_history": ""})
    print(agent_out)


def extract_metadata(df: pd.DataFrame) -> Dict:
    """extracts the meta-data of the given dataframe, including:
    - Num. of columns
    - schema, i.e., column names
    - data types of columns, and
    - an example row."""

    metadata = {
        "Number of Columns": df.shape[1],
        "Schema": df.columns.tolist(),
        "Data Types": str(df.dtypes),
        "Sample": df.head(1).to_dict(orient="records"),
    }

    return metadata


@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


if __name__ == "__main__":
    main()
