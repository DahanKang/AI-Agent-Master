import base64
import json
from openai import OpenAI

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")

client = OpenAI()


def create_page_agent(index: int):
    """
    각 페이지별로 독립적인 Agent를 생성하는 factory 함수
    """

    async def generate_single_image(tool_context: ToolContext):
        print(f"🎨 Generating image for page {index+1}")

        story = tool_context.state.get("storybook_output")

        if story is None:
            return {"status": "error", "message": "storybook_output not found"}

        # pydantic / dict / json 대응
        if hasattr(story, "model_dump"):
            story = story.model_dump()
        elif isinstance(story, str):
            story = json.loads(story)

        pages = story.get("pages", [])

        if index >= len(pages):
            return {"status": "error", "message": f"Page {index} not found"}

        page = pages[index]

        text = page.get("text", "")
        visual = page.get("visual_description", "")

        character = story.get("character_profile", {})

        prompt = f"""
        Children's storybook illustration

        Main character (must be identical in ALL pages):
        - name: {character.get("name")}
        - species: {character.get("species")}
        - appearance: {character.get("appearance")}

        Style:
        - {character.get("style")}
        - consistent across all pages

        STRICT RULES:
        - The character MUST look exactly the same in every image
        - Same face, same clothes, same colors
        - No variation allowed

        Scene:
        {visual}

        Context:
        {text}
        """.strip()

        # 🔥 OpenAI 이미지 생성
        image = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )

        image_base64 = image.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        filename = f"page_{index+1}.png"

        # 🔥 ADK artifact 저장
        artifact = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )

        version = await tool_context.save_artifact(
            filename=filename,
            artifact=artifact
        )

        result = {
            "page": index + 1,
            "text": text,
            "visual": visual,
            "image": filename,
            "version": version
        }

        existing = tool_context.state.get("storybook_images", [])

        if isinstance(existing, str):
            existing = json.loads(existing)

        existing.append(result)

        tool_context.state["storybook_images"] = existing

        # UI에서 filename을 이미지로 추정해 `/dev-ui/<filename>`를 요청하며 404가 나오는 경우가 있어,
        # 실제 이미지(bytes)를 포함한 Part를 직접 반환한다.
        return types.Content(
            parts=[
                artifact,
                types.Part(
                    text=(
                        f"페이지 {result['page']}\n"
                        f"{result['text']}\n"
                        f"(artifact: {result['image']}, version={result['version']})"
                    )
                ),
            ]
        )

    return Agent(
        name=f"IllustratorPage{index+1}",  # 🔥 반드시 고유 이름
        model=MODEL,
        description=f"Generate illustration for page {index+1}",
        instruction=f"""
You are responsible for generating an illustration for page {index+1}.

You MUST:
1. Call the tool to generate the image
2. Do not skip tool usage
3. Return the result clearly
""",
        tools=[generate_single_image],
    )