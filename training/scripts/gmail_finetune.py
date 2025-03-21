#!/usr/bin/env python3
# uv venv
# source .venv/bin/activate
# uv pip install google-api-python-client google-auth-oauthlib pandas
import os
import base64
import pickle
import argparse
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd
import re

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def clean_text(text):
    """Remove email headers, signatures, and clean up the text"""
    # Remove everything from the reply header to the end
    cleaned_text = re.sub(r"\r\n\r\nOn .+?(\r\n)?wrote:.+$", "", text, flags=re.DOTALL)

    # Clean up whitespace
    formatted_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return formatted_text


def get_gmail_service(credentials_path):
    """Get authenticated Gmail API service"""
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)


def fetch_raw_emails(service, max_results=5000):
    """Retrieve raw email data from Gmail"""
    raw_emails = []
    next_page_token = None
    total_fetched = 0

    while total_fetched < max_results:
        # Request messages with pagination
        results = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["SENT"],
                maxResults=min(500, max_results - total_fetched),
                pageToken=next_page_token,
            )
            .execute()
        )

        messages = results.get("messages", [])
        if not messages:
            break

        for i, message in enumerate(messages):
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message["id"], format="full")
                .execute()
            )
            raw_emails.append(msg)
            total_fetched += 1

            if (total_fetched) % 50 == 0:
                print(f"Fetched {total_fetched} emails...")

        # Get the next page token
        next_page_token = results.get("nextPageToken")
        if not next_page_token:
            break

    print(f"Completed fetching {total_fetched} emails")
    return raw_emails


def cmd_fetch(args):
    """Command to fetch raw emails from Gmail"""
    service = get_gmail_service(args.creds)
    print("Fetching raw emails...")
    raw_emails = fetch_raw_emails(service, args.max_results)

    with open(args.outfile, "wb") as f:
        pickle.dump(raw_emails, f)
    print(f"Saved {len(raw_emails)} raw emails to {args.outfile}")


def cmd_transform(args):
    """Command to transform raw emails into structured data"""
    with open(args.infile, "rb") as f:
        raw_emails = pickle.load(f)

    transformed_emails = []
    for i, msg in enumerate(raw_emails):
        payload = msg["payload"]
        headers = payload.get("headers", [])

        subject = next(
            (h["value"] for h in headers if h["name"] == "Subject"), "No Subject"
        )
        to = next((h["value"] for h in headers if h["name"] == "To"), "No Recipient")

        # Extract email body
        if "parts" in payload:
            parts = payload.get("parts", [])
            email_body = ""
            for part in parts:
                if part["mimeType"] == "text/plain":
                    body_data = part["body"].get("data", "")
                    if body_data:
                        email_body = base64.urlsafe_b64decode(body_data).decode("utf-8")
                        break
        else:
            body_data = payload["body"].get("data", "")
            email_body = (
                base64.urlsafe_b64decode(body_data).decode("utf-8") if body_data else ""
            )

        # Skip forwarded messages
        if "---------- Forwarded message ---------" in email_body:
            continue

        # Skip empty messages
        if email_body.strip() == "":
            continue

        # Split email into reply and thread history
        # Most email clients add a delimiter like "On ... wrote:" before quoted text
        split_pattern = r"(?s)On.*wrote"
        parts = re.split(split_pattern, email_body, maxsplit=1)

        if len(parts) > 1:
            reply = clean_text(parts[0])
            thread = clean_text(parts[1])
        else:
            # If there's no split pattern found, treat the whole email as the reply
            reply = clean_text(parts[0])
            thread = ""

        transformed_emails.append(
            {"subject": subject, "to": to, "thread": thread, "reply": reply}
        )

        if (i + 1) % 50 == 0:
            print(f"Transformed {i+1} emails...")

    df = pd.DataFrame(transformed_emails)
    df.to_csv(args.outfile, index=False)
    print(f"Saved {len(transformed_emails)} transformed emails to {args.outfile}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gmail email processing toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch raw emails from Gmail")
    fetch_parser.add_argument(
        "--creds", required=True, help="Path to the credentials.json file"
    )
    fetch_parser.add_argument(
        "--max-results",
        type=int,
        default=5000,
        help="Maximum number of emails to fetch",
    )
    fetch_parser.add_argument(
        "--outfile", default="raw_emails.pkl", help="Output file for raw emails"
    )

    # Transform command
    transform_parser = subparsers.add_parser(
        "transform", help="Transform raw emails into structured data"
    )
    transform_parser.add_argument(
        "--infile", default="raw_emails.pkl", help="Input raw emails file"
    )
    transform_parser.add_argument(
        "--outfile",
        default="transformed_emails.csv",
        help="Output file for transformed emails",
    )

    return parser.parse_args()


def main():
    """Main function to process emails based on command"""
    args = parse_arguments()

    commands = {"fetch": cmd_fetch, "transform": cmd_transform}

    commands[args.command](args)


if __name__ == "__main__":
    main()
