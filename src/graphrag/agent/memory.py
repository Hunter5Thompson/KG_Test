"""
Conversation Memory
===================

Simple buffer to retain conversation history across turns.
"""

from typing import List, Iterable
from langchain_core.messages import BaseMessage


class ConversationMemory:
    """
    Simple conversation buffer with size limit.

    Maintains a sliding window of recent messages to provide
    context continuity while managing memory usage.
    """

    def __init__(self, max_messages: int = 12):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to retain
        """
        self.max_messages = max_messages
        self.history: List[BaseMessage] = []

    def append(self, messages: Iterable[BaseMessage]) -> None:
        """
        Append messages to history and maintain size limit.

        Args:
            messages: Messages to add to history
        """
        self.history.extend(messages)
        # Keep most recent messages within limit
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def snapshot(self) -> List[BaseMessage]:
        """
        Get a copy of the current history.

        Returns:
            List of messages in history
        """
        return list(self.history)

    def clear(self) -> None:
        """Clear all history"""
        self.history.clear()

    def __len__(self) -> int:
        """Get number of messages in history"""
        return len(self.history)
