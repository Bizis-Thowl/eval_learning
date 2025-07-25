SQL_GENERATION_PROMPT = """
The prompt is: {prompt}
The available columns are: {columns}
The table name is: {table_name}
"""


# Construct prompt based on analysis type and data subset
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""


# prompt template for step 2 of tool 3
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""



# prompt template for step 1 of tool 3
FINAL_RESPONSE_PROMPT = """
Generate a final, structured response for the user.
The context is: {messages}
"""



SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
"""