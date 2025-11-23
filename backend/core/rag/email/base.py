# rag/email/base.py
from dataclasses import dataclass
from typing import List, Protocol, Optional
from datetime import datetime


@dataclass
class Email:
    """A normalized representation of an email message."""
    id: str
    sender: str        # e.g. "John Smith <john@example.com>"
    to: str            # e.g. "You <you@gmail.com>"
    subject: str
    body: str
    date: datetime
    provider: str      # e.g. "gmail", "outlook", "imap"


class EmailProvider(Protocol):
    """
    Common interface for any email provider (Gmail, Outlook, generic IMAP, etc.)
    """

    def list_recent_emails(self, limit: int = 50) -> List[Email]:
        """Return the most recent emails, up to 'limit'."""
        ...

    def list_from_sender(self, sender: str, limit: int = 20) -> List[Email]:
        """Return recent emails from a specific sender (by email address or name)."""
        ...

    def get_latest_from_sender(self, sender: str) -> Optional[Email]:
        """Return the most recent email from a specific sender, or None if none."""
        ...