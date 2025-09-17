from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

# 1. Define the LLM (needs OPENAI_API_KEY in your environment)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. Define a dummy tool
def fake_draw_panel(description: str):
    return f"[Generated Image for: {description}]"

tools = [
    Tool(
        name="MangaImageGenerator",
        func=fake_draw_panel,
        description="Generates a manga-style image from a panel description"
    )
]

# 3. Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 4. Run the agent
result = agent.run("Generate 2 panels of a girl scolding a boy about burnt toast.")
print(result)
