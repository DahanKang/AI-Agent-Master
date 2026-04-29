import json
import base64
from openai import OpenAI

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .prompt import ILLUSTRATOR_DESCRIPTION, ILLUSTRATOR_PROMPT

MODEL = LiteLlm(model="openai/gpt-4o")

# OpenAI client
client = OpenAI()


async def generate_storybook_images(tool_context: ToolContext) -> dict:
    print("🔥 IMAGE TOOL CALLED")

    raw_storybook = tool_context.state.get("storybook_output")

    if raw_storybook is None:
        return {"status": "error", "message": "storybook_output not found"}

    # pydantic / dict / json 대응
    if hasattr(raw_storybook, "model_dump"):
        storybook = raw_storybook.model_dump()
    elif isinstance(raw_storybook, dict):
        storybook = raw_storybook
    else:
        storybook = json.loads(raw_storybook)

    pages = storybook.get("pages", [])

    results = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        visual = page["visual_description"]

        prompt = f"""
Create a children's storybook illustration.

Style:
- warm, soft, colorful
- child-friendly
- storybook illustration
- no text in the image

Scene:
{visual}

Context:
{text}
""".strip()

        # 🔥 이미지 생성 (OpenAI)
        image = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )

        # base64 → bytes
        image_base64 = image.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        filename = f"page_{page_number}.png"

        # 🔥 ADK artifact 저장
        artifact = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )

        version = await tool_context.save_artifact(
            filename=filename,
            artifact=artifact
        )

        results.append({
            "page": page_number,
            "text": text,
            "visual": visual,
            "image": filename,
            "version": version
        })

    tool_context.state["storybook_images"] = json.dumps(results, ensure_ascii=False)

    return {
        "status": "success",
        "pages": results
    }


illustrator_agent = Agent(
    name="IllustratorAgent",
    model=MODEL,
    description=ILLUSTRATOR_DESCRIPTION,
    instruction="""
You are IllustratorAgent.

You MUST:
1. Read 'storybook_output' from state
2. Call generate_storybook_images
3. Show result page by page

Format:

Page 1:
Text: ...
Visual: ...
Image: filename

Repeat for all pages.
""",
    tools=[generate_storybook_images],
)