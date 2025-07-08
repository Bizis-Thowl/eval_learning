import pandas as pd
import json
import duckdb
from helper import get_json
from pydantic import BaseModel, Field
from typing import Optional
from prompts.default import (
    # SQL_GENERATION_PROMPT,
    SYSTEM_PROMPT,
    DATA_ANALYSIS_PROMPT,
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_PROMPT,
    FINAL_RESPONSE_PROMPT,
)
from response_models.default import (
    SQLQuery,
    DataAnalysis,
    VisualizationConfig,
    VisualizationCode,
    FinalResponse,
)

from phoenix.client import Client

SQL_GENERATION_PROMPT = Client().prompts.get(prompt_version_id="UHJvbXB0VmVyc2lvbjoy")
my_prompt = {**SQL_GENERATION_PROMPT.format(variables={ "prompt": "test1" })}
SQL_GENERATION_PROMPT = my_prompt["messages"][0]["content"]

import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import os
from opentelemetry.trace import StatusCode


# define the path to the transactional data
TRANSACTION_DATA_FILE_PATH = (
    "../data/Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)


class Agent:
    def __init__(self, client, tool_calling_client, tracer, model: str = "gpt-4o-mini"):
        self.client = client
        self.tool_calling_client = tool_calling_client
        self.tracer = tracer
        self.model = model
        
        # apply decorators to the functions to enable automatic tracing
        self.lookup_sales_data = self.tracer.tool(self.lookup_sales_data)
        self.analyze_sales_data = self.tracer.tool(self.analyze_sales_data)
        self.generate_visualization = self.tracer.tool(self.generate_visualization)
        self.extract_chart_config = self.tracer.chain(self.extract_chart_config)
        self.create_chart = self.tracer.chain(self.create_chart)
        self.handle_tool_calls = self.tracer.chain(self.handle_tool_calls)

        # Define tools/functions that can be called by the model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_sales_data",
                    "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The unchanged prompt that the user provided.",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sales_data",
                    "description": "Analyze sales data to extract insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "The lookup_sales_data tool's output.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The unchanged prompt that the user provided.",
                            },
                        },
                        "required": ["data", "prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_visualization",
                    "description": "Generate Python code to create data visualizations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "The lookup_sales_data tool's output.",
                            },
                            "visualization_goal": {
                                "type": "string",
                                "description": "The goal of the visualization.",
                            },
                        },
                        "required": ["data", "visualization_goal"],
                    },
                },
            },
        ]

        # Dictionary mapping function names to their implementations
        self.tool_implementations = {
            "lookup_sales_data": self.lookup_sales_data,
            "analyze_sales_data": self.analyze_sales_data,
            "generate_visualization": self.generate_visualization,
        }

    # code for step 2 of tool 1
    def generate_sql_query(self, prompt: str, columns: list, table_name: str) -> str:
        """Generate an SQL query based on a prompt"""
        formatted_prompt = SQL_GENERATION_PROMPT.format(
            prompt=prompt, columns=columns, table_name=table_name
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=SQLQuery,
        )

        return response.query

    # code for tool 1
    def lookup_sales_data(self, prompt: str) -> str:
        """Implementation of sales data lookup from parquet file using SQL"""
        try:

            # define the table name
            table_name = "sales"

            # step 1: read the parquet file into a DuckDB table
            df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
            duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

            # step 2: generate the SQL code
            sql_query = self.generate_sql_query(prompt, df.columns, table_name)
            # clean the response to make sure it only includes the SQL code
            sql_query = sql_query.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "")

            # step 3: execute the SQL query
            with self.tracer.start_as_current_span(
                "execute_sql_query", openinference_span_kind="chain"
            ) as span:
                span.set_input(sql_query)
                result = duckdb.sql(sql_query).df()
                span.set_output(value=result)
                span.set_status(StatusCode.OK)

            return result.to_string()
        except Exception as e:
            return f"Error accessing data: {str(e)}"

    # code for tool 2
    def analyze_sales_data(self, prompt: str, data: str) -> str:
        """Implementation of AI-powered sales data analysis"""
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=DataAnalysis,
        )

        analysis = response.analysis
        return analysis if analysis else "No analysis could be generated"

    # code for step 1 of tool 3
    def extract_chart_config(self, data: str, visualization_goal: str) -> dict:
        """Generate chart visualization configuration

        Args:
            data: String containing the data to visualize
            visualization_goal: Description of what the visualization should show

        Returns:
            Dictionary containing line chart configuration
        """
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
            data=data, visualization_goal=visualization_goal
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=VisualizationConfig,
        )

        try:
            # Extract axis and title info from response
            content = response

            # Return structured chart config
            return {
                "chart_type": content.chart_type,
                "x_axis": content.x_axis,
                "y_axis": content.y_axis,
                "title": content.title,
                "data": data,
            }
        except Exception:
            return {
                "chart_type": "line",
                "x_axis": "date",
                "y_axis": "value",
                "title": visualization_goal,
                "data": data,
            }

    # code for step 2 of tool 3
    def create_chart(self, config: dict) -> str:
        """Create a chart based on the configuration"""
        formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=VisualizationCode,
        )

        code = response.code
        code = code.replace("```python", "").replace("```", "")
        code = code.strip()

        return code

    # code for tool 3
    def generate_visualization(self, data: str, visualization_goal: str) -> str:
        """Generate a visualization based on the data and goal"""
        config = self.extract_chart_config(data, visualization_goal)
        code = self.create_chart(config)
        return code

    def generate_final_response(self, messages: list) -> str:
        """Create a final response to the user's question"""
        formatted_prompt = FINAL_RESPONSE_PROMPT.format(messages=messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=FinalResponse,
        )

        return response

    # code for executing the tools returned in the model's response
    def handle_tool_calls(self, tool_calls, messages) -> tuple[list, str]:

        for tool_call in tool_calls:
            function = self.tool_implementations[tool_call.function.name]
            function_args = json.loads(tool_call.function.arguments)
            result = function(**function_args)
            messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )
            print(tool_call.function.name)

        return messages

    def run_agent(self, messages):
        print("Running agent with messages:", messages)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check and add system prompt if needed
        if not any(
            isinstance(message, dict) and message.get("role") == "system"
            for message in messages
        ):
            system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
            messages.append(system_prompt)

        while True:
            # Router Span
            print("Starting router call span")
            with self.tracer.start_as_current_span(
                "router_call",
                openinference_span_kind="chain",
            ) as span:
                span.set_input(value=messages)
                print("Making router call to OpenAI")
                print(messages)
                response = self.tool_calling_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                )
                response_message = response.choices[0].message
                print(get_json(response_message))
                messages.append(get_json(response_message))
                tool_calls = response.choices[0].message.tool_calls
                print("Received response with tool calls:", bool(tool_calls))
                span.set_status(StatusCode.OK)

                # if the model decides to call function(s), call handle_tool_calls
                if tool_calls:
                    print("Starting tool calls span")
                    messages = self.handle_tool_calls(tool_calls, messages)
                    span.set_output(value=tool_calls)
                else:
                    print("No tool calls, returning final response")
                    response = self.generate_final_response(messages)
                    messages.append({"role": "system", "content": response})
                    span.set_output(value=response)
                    return response, messages
                

    # In[35]:

    def start_main_span(self, messages):
        print("Starting main span with messages:", messages)

        with self.tracer.start_as_current_span(
            "AgentRun", openinference_span_kind="agent"
        ) as span:
            span.set_input(value=messages)
            ret = self.run_agent(messages)
            print("Main span completed with return value:", ret)
            span.set_output(value=ret)
            span.set_status(StatusCode.OK)
            return ret


if __name__ == "__main__":
    # print(SQL_GENERATION_PROMPT.messages[0].content)
    my_prompt = {**SQL_GENERATION_PROMPT.format(variables={ "prompt": "test1" })}
    my_prompt = my_prompt["messages"][0]["content"]
    print(my_prompt)