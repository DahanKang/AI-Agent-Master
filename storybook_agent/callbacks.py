from google.adk.agents.callback_context import CallbackContext

def before_agent_callback(callback_context: CallbackContext):
    name = callback_context.agent_name

    if name == "StoryWriterAgent":
        print("📖 스토리 작성 중...")
    elif name == "ParallelIllustratorAgent":
        print("🎨 이미지 생성 시작...")

def after_agent_callback(callback_context: CallbackContext):
    name = callback_context.agent_name

    if name == "StoryWriterAgent":
        print("✅ 스토리 작성 완료")
    elif name == "ParallelIllustratorAgent":
        print("✅ 이미지 생성 완료")