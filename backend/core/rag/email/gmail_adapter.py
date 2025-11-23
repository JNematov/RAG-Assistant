# rag/email/gmail_adapter.py
from typing import List, Optional
from .base import Email
from .imap_adapter import ImapAdapter


class GmailImapAdapter(ImapAdapter):
    """
    Gmail-specific adapter, reusing the generic ImapAdapter logic
    but with Gmail's IMAP host and default settings.

    In practice:
    - host: imap.gmail.com
    - port: 993
    - SSL: True

    You should use:
    - Your full Gmail address as 'username'
    - An app password (recommended) for 'password'
    """

    provider_name = "gmail"

    def __init__(self, username: str, password: str):
        super().__init__(
            host="imap.gmail.com",
            port=993,
            username=username,
            password=password,
            use_ssl=True,
        )