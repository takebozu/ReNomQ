# -*- coding: utf-8 -*-


"""
Exception for errors raised while interpreting nodes.
"""


class NodeException(Exception):
    """Base class for errors raised while interpreting nodes."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
