from google.adk.agents import ParallelAgent
from .single_page_agent import create_page_agent

parallel_illustrator_agent = ParallelAgent(
    name="ParallelIllustratorAgent",
    description="Generate 5 images in parallel",
    sub_agents=[
        create_page_agent(0),
        create_page_agent(1),
        create_page_agent(2),
        create_page_agent(3),
        create_page_agent(4),
    ],
)