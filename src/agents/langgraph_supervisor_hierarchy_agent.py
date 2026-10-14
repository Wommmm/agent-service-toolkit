from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from agents.langgraph_supervisor_agent import add, multiply, web_search
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)

# 多层级 Supervisor 架构


# ┌─────────────────────────────────────────────┐
# │  Main Supervisor (最外层)                    │
# │  管理：research_agent                         │
# │  工具：（自身无工具，靠 handoff）               │
# └──────────────┬──────────────────────────────┘
#                │ handoff
#                ▼
# ┌─────────────────────────────────────────────┐
# │  Research Supervisor (sub-agent)             │
# │  管理：math_agent                             │
# │  工具：web_search                             │
# └──────────────┬──────────────────────────────┘
#                │ handoff
#                ▼
# ┌─────────────────────────────────────────────┐
# │  Math Agent (sub-agent)                      │
# │  工具：add, multiply                          │
# └─────────────────────────────────────────────┘
def workflow(chosen_model):
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # Identify the graph node as a sub-agent
        #你是一名数学专家。始终一次只使用一个工具。
        system_prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    research_agent = (
        create_supervisor(
            [math_agent],
            model=chosen_model,
            tools=[web_search],
            
            # 你是一名世界级研究员，可以使用网络搜索。不要做任何数学运算，你有数学专家来做这个。
            prompt="You are a world class researcher with access to web search. Do not do any math, you have a math expert for that. ",
            supervisor_name="supervisor-research_expert",  # Identify the graph node as a supervisor to the math agent
        )
        .compile(
            name="sub-agent-research_expert"
        )  # Identify the graph node as a sub-agent to the main supervisor
        .with_config(tags=["skip_stream"])
    )  # Stream tokens are ignored for sub-agents in the UI

    # Create supervisor workflow
    return create_supervisor(
        [research_agent],
        model=chosen_model,
        # "你是一名团队主管，管理一位具备数学能力的研究专家。"
        #     "对于当前事件，使用 research_agent。"
        prompt=(
            "You are a team supervisor managing a research expert with math capabilities."
            "For current events, use research_agent. "
        ),
        add_handoff_back_messages=True,
        # UI now expects this to be True so we don't have to guess when a handoff back occurs
        output_mode="full_history",  # otherwise when reloading conversations, the sub-agents' messages are not included
    )  # default name for supervisor is "supervisor".


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
