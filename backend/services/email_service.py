# backend/services/email_service.py

from typing import Dict, Any
import os

from backend.core.rag.email.utils import get_email_provider


def get_latest_email_summary(sender: str) -> Dict[str, Any]:
    email_address = os.environ["EMAIL_ADDRESS"]
    email_password = os.environ["EMAIL_PASSWORD"]
    imap_host = os.environ.get("IMAP_HOST")  # optional for Gmail

    provider = get_email_provider(
        email_address=email_address,
        password=email_password,
        imap_host=imap_host,
    )

    email_obj = provider.get_latest_from_sender(sender)
    if not email_obj:
        return {
            "answer": f"I couldn't find any recent emails from '{sender}'.",
            "sources": [],
        }

    return {
        "answer": f"Found an email from {email_obj.sender} with subject '{email_obj.subject}'.",
        "sources": [
            {
                "sender": email_obj.sender,
                "to": email_obj.to,
                "subject": email_obj.subject,
                "date": email_obj.date.isoformat(),
                "body_preview": email_obj.body[:500],
                "provider": email_obj.provider,
            }
        ],
    }