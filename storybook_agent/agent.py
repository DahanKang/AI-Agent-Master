from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from .sub_agents.story_writer.agent import story_writer_agent
from .sub_agents.illustrator.agent import illustrator_agent

MODEL = LiteLlm(model="openai/gpt-4o")

root_agent = SequentialAgent(
    name="StorybookProducerAgent",
    description="Creates a 5-page children's storybook and then generates illustrations for each page.",
    sub_agents=[
        story_writer_agent,
        illustrator_agent,
    ],
)