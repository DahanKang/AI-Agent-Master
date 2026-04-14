import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, WebSearchTool, set_default_openai_key

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

set_default_openai_key(api_key)

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "life-coach-memory.db",
    )

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


def update_status(status_container, event_type):
    status_messages = {
        "response.web_search_call.in_progress": ("🔍 웹 검색 시작...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색 중...", "running"),
        "response.web_search_call.completed": ("✅ 웹 검색 완료", "complete"),
        "response.completed": (" ", "complete"),
    }

    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)


async def run_agent(user_input):
    agent = Agent(
        name="Life Coach Agent",
        instructions="""
You are a warm, encouraging life coach.

Rules:
- For any user question about motivation, habits, self-improvement, productivity, routines, mindset, or discipline, you must use web search before answering.
- First search the web, then give a practical and encouraging answer.
- Remember the conversation naturally and respond in context.
- Be supportive, clear, and actionable.
- Answer in Korean.
""",
        tools=[WebSearchTool()],
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
    st.subheader("설정")
    if st.button("Reset memory"):
        asyncio.run(session.clear_session())
        st.rerun()

asyncio.run(paint_history())

prompt = st.chat_input("라이프 코치에게 고민을 말해보세요")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    asyncio.run(run_agent(prompt))