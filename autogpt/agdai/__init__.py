""" Autonamous Guidelines Driven AI """
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from autogpt.llm import get_ada_embedding
from .agdai import _agdai_find_similar, _agdai_msg_user
from .agdai_data import ClAgdaiData

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class ClAGDAI(AutoGPTPluginTemplate):

    def __init__(self):
        super().__init__()
        self._name = "AGDAI"
        self._version = "0.1.0"
        self._description = "Autonamous Guidelines Driven AI"
        self._data = ClAgdaiData()

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.
        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model.
        This is called for any call to GPT not only the chat_with_ai() in the main
        loop start_interaction_loop()"""
        pass


    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.
        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return True

    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.
        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return True

    def on_planning(
        self, prompt: PromptGenerator, messages: List[str]
    ) -> Optional[str]:
        """This method is called before the planning chat completeion is done.
        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """
        return self._data.process_msgs(messages)
        # return 'Use the find_similar command defined above to look for similar memories. Ensure your response uses the format specified above.'

    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method. 
        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return True

    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completion is done. The respose is the return from GPT with thoughts strategies etc. Can be used to belay unethical commands
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return self._data.process_actions(response)

    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.
        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    def pre_instruction(self, messages: List[str]) -> List[str]:
        """This method is called before the instruction chat is done.
        Args:
            messages (List[str]): The list of context messages.
        Returns:
            List[str]: The resulting list of messages.
        """
        pass

    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.
        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    def on_instruction(self, messages: List[str]) -> Optional[str]:
        """This method is called when the instruction chat is done.
        Args:
            messages (List[str]): The list of context messages.
        Returns:
            Optional[str]: The resulting message.
        """
        pass

    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.
        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.
        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.
        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.
        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        pass

    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.
        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.
        Args:
            command_name (str): The command name.
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """This method is called to check that the plugin can
        handle the chat_completion method. Called directly before request to GPT, not only in main loop
        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
        Returns:
            bool: True if the plugin can handle the chat_completion method."""
        return False

    def handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """This method is called when the chat completion is done.
        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
        Returns:
            str: The resulting response.
        """
        return None

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """

        self._data.set_agent_name(prompt.name)
        prompt.add_command(
            "find_similar",
            "Find similar memories that succeeded in the past. ",
            {"memory": "<memory_like_this>"},
            _agdai_find_similar,
        )
        prompt.add_command(
            'telegram_message_user',
            'Message user',
            {"message": "<message_to_send>"},
            _agdai_msg_user,
        )
        return prompt
