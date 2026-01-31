import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import os

# Cloud LLM (Groq)
for key in ["GROQ_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1,
)

# Wrapped search tool
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time signals."""
    return DuckDuckGoSearchRun().run(query)
# Agents
researcher = Agent(
    role="Researcher",
    goal="Research AI trends",
    backstory="Expert analyst — find 10-15 real 2026 signals/quotes with sources",
    tools=[duckduckgo_search],
    llm=llm,
    verbose=False
)

writer = Agent(
    role="Writer",
    goal="Draft LinkedIn posts",
    backstory="Copywriter — output ONLY clean posts",
    llm=llm,
    verbose=False
)

coder = Agent(
    role="Coder",
    goal="Write code",
    backstory="Python engineer — embed market signals in comments",
    llm=llm,
    verbose=False
)

st.title("🚀 Research Crew Demo")
st.write("Daily swarm for research, posts, code — powered by CrewAI + Groq (cloud)")

st.info("🤖 Cloud-powered — no local setup needed")

goal = st.text_area("Enter goal (e.g., AI trends 2026)", height=100)

if st.button("Run Research Crew"):
    if goal.strip():
        with st.spinner("Crew running (30-90 seconds)..."):
            task1 = Task(
                description=f"Research: {goal}\nFind 10-15 real 2026 market signals/quotes with sources",
                expected_output="Signals list with sources",
                agent=researcher
            )
            task2 = Task(
                description="Write 2 LinkedIn posts",
                expected_output="Two clean posts",
                agent=writer,
                context=[task1]
            )
            task3 = Task(
                description=f"Write simple code example for: {goal}\nEmbed 5-8 market signals from research in comments",
                expected_output="Short Python code with signals",
                agent=coder,
                context=[task1]
            )

            crew = Crew(agents=[researcher, writer, coder], tasks=[task1, task2, task3], verbose=False)
            result = crew.kickoff()

            st.success("Complete!")
            st.markdown(str(result))

            st.download_button(
                label="Download Output",
                data=str(result),
                file_name="research_crew_output.txt",
                mime="text/plain"
            )
    else:
        st.warning("Enter a goal")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Usage Tips")
    st.markdown("- Keep goal <100 words")
    st.markdown("- Specific topics best")
    st.markdown("- Unlimited runs (cloud)")
    
    st.markdown("---")
    st.markdown("### 💡 Features")
    st.markdown("- Real-time research signals")
    st.markdown("- LinkedIn posts")
    st.markdown("- Code examples")
    
    st.markdown("---")
    st.markdown("Feedback — DM @mike51802 on X")

st.write("Demo by Mike Millard — building AI assets")
