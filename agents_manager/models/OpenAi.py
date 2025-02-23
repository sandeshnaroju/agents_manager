from typing import Iterator
from typing import List, Dict, Any, Union, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from agents_manager.Model import Model


class OpenAi(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the OpenAi model with a name and optional keyword arguments.

        Args:
            name (str): The name of the OpenAI model (e.g., "gpt-3.5-turbo").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        if name is None:
            raise ValueError("A valid  OpenAI model name is required")

        super().__init__(name, **kwargs)

        self.client = OpenAI(
            api_key=kwargs.get("api_key"),  # type: Optional[str]
        )

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the OpenAI model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        # remove api_key from kwargs
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")

        response = self.client.chat.completions.create(
            model=self.name,  # type: str
            messages=self.get_messages(),  # type: List[Dict[str, str]]
            response_format={"type": "text"},  # type: Dict[str, str]
            temperature=self.kwargs.get("temperature", 1),  # type: float
            max_completion_tokens=self.kwargs.get("max_completion_tokens", 15688),  # type: Optional[int]
            top_p=self.kwargs.get("top_p", 1),  # type: float
            stream=self.kwargs.get("stream", False),  # type: bool
            **self.kwargs.get("penalties", {"frequency_penalty": 0, "presence_penalty": 0}),  # type: Dict[str, float]
            **self.kwargs  # type: Dict[str, Any]
        )
        message = response.choices[0].message
        return {
            "tool_calls": message.tool_calls,
            "content": message.content,
        }

    def generate_response_stream(self) -> Iterator[ChatCompletionChunk]:
        """
        Generate a streaming response from the OpenAI model.

        Returns:
            Iterator[ChatCompletionChunk]: An iterator over ChatCompletionChunk objects.
        """
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")

        return self.client.chat.completions.create(
            model=self.name,  # type: str
            messages=self.get_messages(),  # type: List[Dict[str, str]]
            response_format={"type": "text"},  # type: Dict[str, str]
            temperature=self.kwargs.get("temperature", 1),  # type: float
            max_completion_tokens=self.kwargs.get("max_completion_tokens", 15688),  # type: Optional[int]
            top_p=self.kwargs.get("top_p", 1),  # type: float
            stream=True,  # type: bool
            **self.kwargs.get("penalties", {"frequency_penalty": 0, "presence_penalty": 0}),  # type: Dict[str, float]
            **self.kwargs  # type: Dict[str, Any]
        )

    def get_tool_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
                "strict": True,
            }
        }

    def get_tool_output_format(self) -> Dict[str, Any]:
        return {
            "id": "{id}",
            "type": "function",
            "function": {
                "name": "{name}",
                "arguments": "{arguments}"
            }
        }
