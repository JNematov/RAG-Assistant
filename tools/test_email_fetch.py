# test_email_fetch.py (for you to experiment with)
import os
from backend.core.rag.email.utils import get_email_provider

def main():
    email_address = os.environ["EMAIL_ADDRESS"]
    email_password = os.environ["EMAIL_PASSWORD"]
    imap_host = os.environ.get("IMAP_HOST")  # needed for non-gmail

    provider = get_email_provider(
        email_address=email_address,
        password=email_password,
        imap_host=imap_host,
    )

    # 1) List recent emails
    recent = provider.list_recent_emails(limit=5)
    print("\nRecent emails:")
    for e in recent:
        print(f"- [{e.date}] {e.sender} -> {e.subject}")

    # 2) Latest from some sender
    latest_from_john = provider.get_latest_from_sender("John")
    if latest_from_john:
        print("\nLatest email from John:")
        print(f"Subject: {latest_from_john.subject}")
        print(f"Body:\n{latest_from_john.body[:500]}...")
    else:
        print("\nNo emails found from John.")

if __name__ == "__main__":
    main()

    # jd7356392
    # SomeRandomAhhPasswordIJustCameUpWith