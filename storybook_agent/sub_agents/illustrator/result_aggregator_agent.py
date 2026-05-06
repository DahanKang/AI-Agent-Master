import json
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")


async def aggregate_results(tool_context: ToolContext):
    # state에서 결과 가져오기
    results = tool_context.state.get("storybook_images", [])

    if isinstance(results, str):
        results = json.loads(results)

    if not results:
        return "No images generated."

    # 정렬
    results = sorted(results, key=lambda x: x["page"])

    # 출력 리스트
    output_parts = []

    for r in results:
        artifact = await tool_context.load_artifact(
            r["image"],
        )

        if artifact is not None:
            output_parts.append(artifact)  # 이미지
        else:
            output_parts.append(
                types.Part(
                    text=f"(이미지 로드 실패) {r['image']} (version={r.get('version')})"
                )
            )

        output_parts.append(
            types.Part(
                text=f"페이지 {r['page']}\n{r['text']}"
            )
        )

    return types.Content(parts=output_parts)


result_aggregator_agent = Agent(
    name="ResultAggregator",
    model=MODEL,
    instruction="Collect and display final storybook in correct order",
    tools=[aggregate_results],
)