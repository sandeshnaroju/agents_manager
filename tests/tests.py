import json
from agents_manager import AgentManager
from tree_agents import (
    agent1,
    agent2,
    agent3,
    handover_agent3,
    transfer_to_agent2,
)
from chain_agents import agent6, agent4, agent5, transfer_to_agent5, handover_agent6

STORY = """
A quiet seed fell into rich soil.
Rain came gently, and the sun followed.
Days passed. A sprout emerged, green and hopeful.
It grew tall, touched by breeze and birdsong.
In time, it became a tree, offering shade and shelter.
Life continued, simple and still, beneath its patient branches.
"""


def test_tree_handover():
    manager = AgentManager()

    agent1.tools = [transfer_to_agent2, handover_agent3]

    manager.add_agent(agent1)
    manager.add_agent(agent2)
    manager.add_agent(agent3)

    resp = manager.run_agent(
        "agent1",
        [{"role": "user", "content": f"Summarize it and then extend it {STORY}"}],
    )

    resp = json.loads(resp["content"])

    assert resp["summarize"]["pos"] == 1
    assert resp["extend"]["pos"] == 2


def test_chain_handover():
    manager = AgentManager()

    agent4.tools = [transfer_to_agent5]
    agent5.tools = [handover_agent6]

    manager.add_agent(agent4)
    manager.add_agent(agent5)
    manager.add_agent(agent6)

    resp = manager.run_agent(
        "agent4",
        [{"role": "user", "content": "Give me the secret"}],
    )

    resp = json.loads(resp["content"])

    assert resp["secret"] == "chaining_agents_works"
    assert resp["tool_name"] == "handover_agent6"
