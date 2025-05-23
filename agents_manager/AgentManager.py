import json
from typing import List, Optional, Any, Generator, Dict, Callable

from agents_manager.Container import Container
from agents_manager.Agent import Agent


class AgentManager:
    def __init__(self) -> None:
        """
        Initialize the AgentManager with an empty list of agents.
        """
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the manager's list.

        Args:
            agent (Agent): The agent instance to add.
        """
        if not isinstance(agent, Agent):
            raise ValueError("Only Agent instances can be added")
        _, existing_agent = self.get_agent(agent.name)
        if not existing_agent:
            self.agents.append(agent)

    def get_agent(self, name: str) -> tuple[Optional[int], Optional[Agent]]:
        """
        Retrieve an agent by name.
        Args:
            name (str): The name of the agent to find.
        Returns:
            tuple[Optional[int], Optional[Agent]]: A tuple containing the index and agent if found, else (None, None).
        """

        for _, agent in enumerate(self.agents):
            if agent.name == name:
                return _, agent
        return None, None

    def _initialize_user_input(
        self, name: str, user_input: Optional[Any] = None
    ) -> tuple[Optional[int], Optional[Agent]]:

        _, agent = self.get_agent(name)

        if agent is None:
            raise ValueError(f"No agent found with name: {name}")
        agent.set_messages([])
        agent.set_system_message(agent.instruction)
        agent.set_tools(agent.tools)
        agent.set_output_format()
        if user_input:
            agent.set_user_message(user_input)
        return _, agent

    @staticmethod
    def _prepare_final_messages(
        agent: Agent, current_messages: list, tool_responses: list
    ):
        tool_response = agent.get_model().get_tool_message(tool_responses)
        if isinstance(tool_response, dict):
            current_messages.append(tool_response)
        if isinstance(tool_response, list):
            current_messages.extend(tool_response)
        agent.set_messages(current_messages)

    def run_agent(self, name: str, user_input: Optional[Any] = None) -> Dict:
        """
        Run a specific agent's non-streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (str, optional): Additional user input to append to messages.

        Returns:
            Any: The agent's response.
        """
        _, agent = self._initialize_user_input(name, user_input)
        response = agent.get_response()
        if not response["tool_calls"]:
            return response

        tool_calls = response["tool_calls"]

        current_messages = agent.get_messages()
        assistant_message = agent.get_model().get_assistant_message(response)

        if isinstance(assistant_message, dict):
            current_messages.append(assistant_message)
        if isinstance(assistant_message, list):
            current_messages.extend(assistant_message)

        tool_responses = []

        for tool_call in tool_calls:
            output = agent.get_model().get_keys_in_tool_output(tool_call)
            id, function_name = output["id"], output["name"]
            arguments = (
                json.loads(output["arguments"])
                if isinstance(output["arguments"], str)
                else output["arguments"]
            )

            for tool in agent.tools:
                if isinstance(tool, Callable) and (
                    tool.__name__ == function_name
                    and not tool.__name__.startswith("handover_")
                ):
                    tool_result = tool(**arguments)

                    if isinstance(tool_result, Agent):
                        if not self.get_agent(tool_result.name)[1]:
                            self.add_agent(tool_result)
                        child_response = self.run_agent(tool_result.name, user_input)
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(
                                    child_response.get("content", child_response)
                                ),
                                "name": function_name,
                            }
                        )
                    else:
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )

                elif isinstance(tool, Callable) and (
                    tool.__name__.startswith("handover_")
                    and tool.__name__ == function_name
                ):
                    tool_result = tool()
                    child_response = self.run_agent(tool_result, user_input)
                    tool_responses.append(
                        {
                            "id": id,
                            "tool_result": str(
                                child_response.get("content", child_response)
                            ),
                            "name": function_name,
                        }
                    )

                elif isinstance(tool, Container) and (
                    tool.name == function_name and not tool.name.startswith("handover_")
                ):
                    tool_result = tool.run(arguments)

                    if isinstance(tool_result, Agent):
                        if not self.get_agent(tool_result.name)[1]:
                            self.add_agent(tool_result)
                        child_response = self.run_agent(tool_result.name, user_input)
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(
                                    child_response.get("content", child_response)
                                ),
                                "name": function_name,
                            }
                        )
                    else:
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )

        self._prepare_final_messages(agent, current_messages, tool_responses)
        response = agent.get_response()

        if not response["tool_calls"]:
            return response

    def run_agent_stream(
        self,
        name: str,
        user_input: Optional[Any] = None,
    ) -> Generator[Dict, None, None]:
        """
        Run a specific agent's streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (str, optional): Additional user input to append to messages.

        Returns:
            Any: The agent's response.
        """
        position, agent = self._initialize_user_input(name, user_input)
        initial_tools = agent.get_tools()
        if not initial_tools and position == 0:
            yield from agent.get_stream_response()
            return

        response = agent.get_response()
        if not response["tool_calls"]:
            return response["content"]

        tool_calls = response["tool_calls"]
        current_messages = agent.get_messages()
        assistant_message = agent.get_model().get_assistant_message(response)
        if isinstance(assistant_message, dict):
            current_messages.append(assistant_message)
        if isinstance(assistant_message, list):
            current_messages.extend(assistant_message)
        tool_responses = []
        for tool_call in tool_calls:
            output = agent.get_model().get_keys_in_tool_output(tool_call)
            id, function_name = output["id"], output["name"]
            arguments = (
                json.loads(output["arguments"])
                if isinstance(output["arguments"], str)
                else output["arguments"]
            )

            for tool in agent.tools:
                if isinstance(tool, Callable) and (
                    tool.__name__ == function_name
                    and not tool.__name__.startswith("handover_")
                ):
                    tool_result = tool(**arguments)

                    if isinstance(tool_result, Agent):
                        if not self.get_agent(tool_result.name)[1]:
                            self.add_agent(tool_result)
                        child_response = self.run_agent(tool_result.name, user_input)

                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(
                                    child_response.get("content", child_response)
                                ),
                                "name": function_name,
                            }
                        )
                    else:
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )

                elif isinstance(tool, Callable) and (
                    tool.__name__.startswith("handover_")
                    and tool.__name__ == function_name
                ):
                    tool_result = tool()
                    child_response = self.run_agent(tool_result, user_input)
                    tool_responses.append(
                        {
                            "id": id,
                            "tool_result": str(
                                child_response.get("content", child_response)
                            ),
                            "name": function_name,
                        }
                    )

                elif isinstance(tool, Container) and (
                    tool.name == function_name and not tool.name.startswith("handover_")
                ):
                    tool_result = tool.run(arguments)

                    if isinstance(tool_result, Agent):
                        if not self.get_agent(tool_result.name)[1]:
                            self.add_agent(tool_result)
                        child_response = self.run_agent(tool_result.name, user_input)

                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(
                                    child_response.get("content", child_response)
                                ),
                                "name": function_name,
                            }
                        )
                    else:
                        tool_responses.append(
                            {
                                "id": id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )
        self._prepare_final_messages(agent, current_messages, tool_responses)
        yield from agent.get_stream_response()
        return
