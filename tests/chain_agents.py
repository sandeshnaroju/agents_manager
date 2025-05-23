from pydantic import BaseModel
from agents_manager import Agent
from agents_manager.models import OpenAi


class Format(BaseModel):
    secret: str


def transfer_to_agent5() -> Agent:
    """Follow me for success"""
    return agent5


def handover_agent6() -> str:
    """Has some secrets"""
    return "agent6"


openai_model = OpenAi(
    name="gpt-4o-mini",
)

agent4 = Agent(
    name="agent4",
    instruction='Your only task is giving the secret key to the user using proper tools. Response will just be this dict nothing else {"secret": <secret_key>, "tool_name": <tool_name>}',
    model=openai_model,
    output_format=Format,
)

agent5 = Agent(
    name="agent5",
    instruction="Use tools to find secret key for user. After you find just say `here is the secret key: <secret_key>. I got it from tool named <tool_name>` and nothing else",
    model=openai_model,
)

agent6 = Agent(
    name="agent6",
    instruction="If someone asks for secret key give them this `chaining_agents_works`",
    model=openai_model,
)
