import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool, set_default_openai_key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

set_default_openai_key(api_key)
client = OpenAI(api_key=api_key)

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "life-coach-memory.db",
    )

if "vector_store_id" not in st.session_state:
    # 이미 만든 Vector Store ID가 있다면 .env에 넣어서 재사용
    st.session_state["vector_store_id"] = os.getenv("VECTOR_STORE_ID")

session = st.session_state["session"]


async def paint_history():
    items = await session.get_items()

    for item in items:
        if item.get("role") == "user":
            with st.chat_message("user"):
                content = item.get("content")
                if isinstance(content, str):
                    st.write(content)

        elif item.get("role") == "assistant":
            with st.chat_message("assistant"):
                if item.get("type") == "message":
                    st.write(item["content"][0]["text"])

        elif item.get("type") == "web_search_call":
            with st.chat_message("assistant"):
                st.write("🔍 웹 검색 사용")
        
        elif item.get("type") == "file_search_call":
            with st.chat_message("assistant"):
                st.write("🗂️ 목표/일기 문서 검색 사용")


def update_status(status_container, event_type):
    status_messages = {
        "response.web_search_call.in_progress": ("🔍 웹 검색 시작...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색 중...", "running"),
        "response.web_search_call.completed": ("✅ 웹 검색 완료", "complete"),
        "response.file_search_call.in_progress": ("🗂️ 문서 검색 시작...", "running"),
        "response.file_search_call.searching": ("🗂️ 문서 검색 중...", "running"),
        "response.file_search_call.completed": ("✅ 문서 검색 완료", "complete"),
        "response.completed": (" ", "complete"),
    }

    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)

def ensure_vector_store():
    if not st.session_state["vector_store_id"]:
        vector_store = client.vector_stores.create(name="life-coach-goals-and-journal")
        st.session_state["vector_store_id"] = vector_store.id
    return st.session_state["vector_store_id"]


def upload_file_to_vector_store(uploaded_file):
    vector_store_id = ensure_vector_store()

    openai_file = client.files.create(
        file=(uploaded_file.name, uploaded_file.getvalue()),
        purpose="user_data",
    )

    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=openai_file.id,
    )

    return vector_store_id, openai_file.id

async def run_agent(user_input):
    tools = [WebSearchTool()]

    if st.session_state["vector_store_id"]:
        tools.append(
            FileSearchTool(
                vector_store_ids=[st.session_state["vector_store_id"]],
                max_num_results=3,
            )
        )

    agent = Agent(
        name="Life Coach Agent",
        instructions="""
You are a warm, encouraging life coach.

Rules:
- Remember the conversation and respond in context.
- When the user asks about their goals, habits, progress, routines, diary, journal, or personal plans, first search the uploaded files.
- After checking the user's files, use web search when helpful to find practical, evidence-based, or current advice.
- Combine the user's personal goals/history with web findings to give personalized recommendations.
- Track progress over time by referring to prior conversation and uploaded documents.
- Be supportive, clear, and practical.
- Answer in Korean.
""",
        tools=tools,
    )

    with st.chat_message("assistant"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        full_response = ""

        stream = Runner.run_streamed(
            agent,
            user_input,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    full_response += event.data.delta
                    text_placeholder.write(full_response)


st.title("Life Coach Agent")

with st.sidebar:
    st.subheader("목표 / 일기 문서 업로드")

    uploaded_files = st.file_uploader(
        "PDF 또는 TXT 파일 업로드",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.status(f"{uploaded_file.name} 업로드 중...") as status:
                vector_store_id, file_id = upload_file_to_vector_store(uploaded_file)
                status.update(
                    label=f"✅ {uploaded_file.name} 업로드 완료",
                    state="complete",
                )
        st.success(f"현재 Vector Store ID: {st.session_state['vector_store_id']}")

    if st.button("Reset memory"):
        asyncio.run(session.clear_session())
        st.rerun()

asyncio.run(paint_history())

prompt = st.chat_input("라이프 코치에게 고민을 말해보세요")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    asyncio.run(run_agent(prompt))