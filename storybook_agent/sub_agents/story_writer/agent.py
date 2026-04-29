from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .prompt import STORY_WRITER_DESCRIPTION, STORY_WRITER_PROMPT
from pydantic import BaseModel, Field
from typing import List


class StoryPageOutput(BaseModel):
    page_number: int = Field(description="Page number from 1 to 5")
    text: str = Field(description="Story text for this page")
    visual_description: str = Field(description="Illustration description for this page")


class StoryBookOutput(BaseModel):
    title: str = Field(description="Title of the storybook")
    theme: str = Field(description="Theme of the storybook")
    pages: List[StoryPageOutput] = Field(description="Exactly 5 story pages")


MODEL = LiteLlm(model="openai/gpt-4o")

story_writer_agent = Agent(
    name="StoryWriterAgent",
    description=STORY_WRITER_DESCRIPTION,
    instruction=STORY_WRITER_PROMPT,
    model=MODEL,
    output_schema=StoryBookOutput,
    output_key="storybook_output",
)