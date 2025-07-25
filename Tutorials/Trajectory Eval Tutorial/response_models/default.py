from pydantic import BaseModel, Field
from typing import Optional

# ### Tool 1: Retrieve SQL Data
class SQLQuery(BaseModel):
    query: str = Field(..., description="The SQL query to execute")


# ### Tool 2: Analyze Data
class DataAnalysis(BaseModel):
    analysis: str = Field(..., description="The analysis of the data")
    
    
# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")
    

class VisualizationCode(BaseModel):
    code: str = Field(..., description="The python code to generate the visualization. Use the data from the config")


class FinalResponse(BaseModel):
    code: Optional[str] = Field(None, description="The python code to execute")
    observed_trends: Optional[str] = Field(None, description="The observed trends in the data")
    natural_language_response: Optional[str] = Field(None, description="The natural language response to the user's question")