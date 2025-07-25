from openai import OpenAI
import instructor
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from helper import get_openai_api_key, get_phoenix_endpoint


def init_phoenix(model: str = "gpt-4o-mini", project_name: str = "tracing-agent"):
    # initialize the OpenAI client
    openai_api_key = get_openai_api_key()
    client = instructor.from_openai(OpenAI())
    tool_calling_client = OpenAI(api_key=openai_api_key)

    MODEL = model

    PROJECT_NAME = project_name

    PHOENIX_ENDPOINT = get_phoenix_endpoint()


    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint= PHOENIX_ENDPOINT + "v1/traces",
        
    )

    OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)

    tracer = tracer_provider.get_tracer(__name__)
    
    return client, tool_calling_client, tracer