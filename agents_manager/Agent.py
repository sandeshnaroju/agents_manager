from typing import List, Any, Optional, Callable, Dict

from agents_manager.Model import Model
from agents_manager.utils import function_to_json


class Agent:
    def __init__(self, name: Optional[str] = None,
                 instruction: Optional[str] = None,
                 model: Optional[Model] = None,
                 tools: Optional[List[Callable]] = None,
                 tool_choice: Optional[Callable] = None) -> None:
        """
        Initialize the Agent with a name, instruction, tools, and a model instance.

        Args:
            name (str, optional): The name of the agent.
            instruction (str, optional): The system instruction for the agent.
            tools (List[Callable], optional): List of callable tools the agent can use.
            model (Model, optional): An instance of a concrete Model subclass.
        """
        self.name: Optional[str] = name
        self.instruction: str = instruction or ""
        self.tools: List[Callable] = tools or []
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid instance of a Model subclass is required")
        self.model: Model = model
        self.tool_choice = tool_choice

    def set_instruction(self, instruction: str) -> None:
        """
        Set the system instruction and update the model's messages.

        Args:
            instruction (str): The system instruction for the agent.
        """
        self.instruction = instruction

    def get_instruction(self) -> str:
        """
        Get the system instruction for the agent.

        Returns:
            str: The system instruction.
        """
        return self.instruction

    def get_messages(self) -> Optional[List[Dict[str, str]]]:
        """
        Get the messages for the model.

        Returns:
            Optional[List[Dict[str, str]]]: The list of message dictionaries if set, else None.
        """
        return self.model.get_messages()

    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the messages for the model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".
        """
        self.model.set_messages(messages)

    def set_tools(self, tools: List[Callable]) -> None:
        """
        Set the tools for the agent and update the model's kwargs.

        Args:
            tools (List[Callable]): List of callable tools to be used by the agent.
        """
        self.tools = tools
        json_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, Callable):
                json_tools.append(function_to_json(tool, self.model.get_tool_format()))
        self.model.set_kwargs({
            "tools": json_tools
        })

    def get_tools(self) -> List[Callable]:
        """
        Get the tools for the agent.

        Returns:
            List[Callable]: The list of callable tools.
        """
        return self.tools

    def get_model(self) -> Model:
        """
        Get the model instance for the agent.

        Returns:
            Model: The model instance.
        """
        return self.model

    def set_model(self, model: Model) -> None:
        """
        Set the model instance for the agent.

        Args:
            model (Model): An instance of a concrete Model subclass.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid instance of a Model subclass is required")
        self.model = model

    def set_tool_choice(self, tool_choice: Callable) -> None:
        """
        Set the tool choice function for the agent.

        Args:
            tool_choice (Callable): The function that selects a tool from the list of tools.
        """
        self.tool_choice = tool_choice
        self.model.set_kwargs({
            "tool_choice": function_to_json(tool_choice)
        })

    def get_response(self) -> Any:
        """
        Generate a non-streaming response from the model.

        Returns:
            Any: The response, type depends on the model's implementation.
        """
        if not hasattr(self.model, 'messages') or self.model.messages is None:
            raise ValueError("Messages must be set before generating a response")
        return self.model.generate_response()

    def get_response_stream(self) -> Any:
        """
        Generate a streaming response from the model.

        Returns:
            Any: The streaming response, type depends on the model's implementation.
        """
        if not hasattr(self.model, 'messages') or self.model.messages is None:
            raise ValueError("Messages must be set before generating a stream")
        return self.model.generate_response_stream()
