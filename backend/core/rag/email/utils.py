# rag/email/utils.py
from typing import Literal
from .gmail_adapter import GmailImapAdapter
from .imap_adapter import ImapAdapter

ProviderName = Literal["gmail", "outlook", "yahoo", "imap"]


def detect_provider(email_address: str) -> ProviderName:
    """
    Very simple domain-based detection.

    - *@gmail.com / @googlemail.com -> "gmail"
    - *@outlook.com / @hotmail.com / @live.com -> "outlook"
    - *@yahoo.com / @ymail.com -> "yahoo"
    - anything else -> "imap" (generic fallback)
    """
    try:
        domain = email_address.split("@", 1)[1].lower()
    except IndexError:
        # invalid email format; default to IMAP
        return "imap"

    if domain in ("gmail.com", "googlemail.com"):
        return "gmail"
    elif domain in ("outlook.com", "hotmail.com", "live.com"):
        return "outlook"
    elif domain in ("yahoo.com", "ymail.com"):
        return "yahoo"
    else:
        return "imap"


def get_email_provider(
    email_address: str,
    password: str,
    imap_host: str | None = None,
) -> object:
    """
    Return the appropriate EmailProvider instance based on the email address domain.

    - For Gmail: uses GmailImapAdapter with host=imap.gmail.com
    - For others: uses ImapAdapter, requires 'imap_host' (e.g. imap.yourcompany.com)

    NOTE: In a real app, you'd likely fetch 'password' and 'imap_host'
    from environment variables or a config file, not pass them directly.
    """
    provider_name = detect_provider(email_address)

    if provider_name == "gmail":
        # Gmail IMAP server host is fixed
        return GmailImapAdapter(username=email_address, password=password)
    else:
        if imap_host is None:
            raise ValueError("imap_host must be provided for non-Gmail IMAP accounts")
        return ImapAdapter(host=imap_host, username=email_address, password=password)