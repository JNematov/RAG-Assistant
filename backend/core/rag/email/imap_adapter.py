# rag/email/imap_adapter.py
import imaplib
import ssl
from typing import List, Optional
from datetime import datetime

from email import message_from_bytes
from email.message import Message
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime

from .base import Email, EmailProvider


class ImapAdapter:
    """
    Generic IMAP-based adapter.

    Uses:
    - Python's 'imaplib' to connect to an IMAP server
    - 'email' module to parse messages

    This will work with:
    - Gmail's IMAP (with correct host + app password)
    - Many other IMAP providers (with correct host and credentials)
    """

    provider_name = "imap"

    def __init__(self, host: str, username: str, password: str, port: int = 993, use_ssl: bool = True):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl

    # ---------- Connection helpers ----------

    def _connect(self) -> imaplib.IMAP4:
        """
        Connect and log into the IMAP server.

        Returns an IMAP4/IMAP4_SSL instance.
        """
        if self.use_ssl:
            context = ssl.create_default_context()
            imap = imaplib.IMAP4_SSL(self.host, self.port, ssl_context=context)
        else:
            imap = imaplib.IMAP4(self.host, self.port)

        imap.login(self.username, self.password)
        return imap

    def _fetch_message(self, imap: imaplib.IMAP4, msg_id: bytes) -> Optional[Email]:
        """
        Fetch a single message by IMAP id and parse into an Email object.
        """
        typ, data = imap.fetch(msg_id, "(RFC822)")
        if typ != "OK" or not data or not data[0]:
            return None

        raw = data[0][1]
        msg: Message = message_from_bytes(raw)

        # Helper to decode headers that may be encoded
        def decode_str(value: str | None) -> str:
            if not value:
                return ""
            try:
                dh = decode_header(value)
                return str(make_header(dh))
            except Exception:
                return value

        subject = decode_str(msg.get("Subject"))
        from_ = decode_str(msg.get("From"))
        to = decode_str(msg.get("To"))

        # Parse date
        date_header = msg.get("Date")
        try:
            date = parsedate_to_datetime(date_header) if date_header else datetime.utcnow()
        except Exception:
            date = datetime.utcnow()

        # Extract plain text body (very simple version)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition") or "").lower()
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body_bytes = part.get_payload(decode=True) or b""
                        body = body_bytes.decode(part.get_content_charset() or "utf-8", errors="ignore")
                        break
                    except Exception:
                        continue
        else:
            # Non-multipart: just decode the payload
            try:
                body_bytes = msg.get_payload(decode=True) or b""
                body = body_bytes.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                body = msg.get_payload()

        # msg_id is bytes; convert to str
        msg_id_str = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else str(msg_id)

        return Email(
            id=msg_id_str,
            sender=from_ or "",
            to=to or "",
            subject=subject or "",
            body=body or "",
            date=date,
            provider=self.provider_name,
        )

    # ---------- Public methods implementing EmailProvider ----------

    def list_recent_emails(self, limit: int = 50) -> List[Email]:
        """
        Return the most recent 'limit' emails from the INBOX.
        """
        imap = self._connect()
        try:
            # Select the INBOX
            typ, _ = imap.select("INBOX")
            if typ != "OK":
                return []

            # Search all message IDs
            typ, data = imap.search(None, "ALL")
            if typ != "OK" or not data or not data[0]:
                return []

            # data[0] contains space-separated ids
            all_ids = data[0].split()
            # Take the last 'limit' ids (most recent)
            recent_ids = all_ids[-limit:]

            emails: List[Email] = []
            for msg_id in reversed(recent_ids):  # reversed -> newest first
                email_obj = self._fetch_message(imap, msg_id)
                if email_obj:
                    emails.append(email_obj)

            return emails
        finally:
            try:
                imap.close()
            except Exception:
                pass
            imap.logout()

    def list_from_sender(self, sender: str, limit: int = 20) -> List[Email]:
        """
        Return recent emails from a specific sender.

        'sender' can be an email address or part of the "From" header.
        """
        imap = self._connect()
        try:
            typ, _ = imap.select("INBOX")
            if typ != "OK":
                return []

            # Search for messages FROM this sender.
            # NOTE: Depending on server, you may need exact email, e.g. "john@example.com"
            search_criteria = f'(FROM "{sender}")'
            typ, data = imap.search(None, search_criteria)
            if typ != "OK" or not data or not data[0]:
                return []

            matched_ids = data[0].split()
            # Take last 'limit' ids
            target_ids = matched_ids[-limit:]

            emails: List[Email] = []
            for msg_id in reversed(target_ids):
                email_obj = self._fetch_message(imap, msg_id)
                if email_obj:
                    emails.append(email_obj)

            return emails
        finally:
            try:
                imap.close()
            except Exception:
                pass
            imap.logout()

    def get_latest_from_sender(self, sender: str) -> Optional[Email]:
        """
        Return the most recent email from a specific sender, or None if not found.
        """
        emails = self.list_from_sender(sender, limit=10)
        if not emails:
            return None

        # Emails are already newest-first from list_from_sender
        return emails[0]