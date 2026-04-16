"""PDF delivery: relay download URL + transactional email.

Two URL strategies (auto-selected):

Strategy A — Server relay (preferred, no R2 S3 creds needed):
    Generates a time-limited HMAC-signed token and emails a link to
    GET /api/playbook/download/{token}.  The download endpoint proxies
    the PDF from R2 using the Cloudflare API token.

    Required env vars:
        CLOUDFLARE_API_TOKEN    — CF API token (with R2 object read)
        CF_ACCOUNT_ID           — Cloudflare account ID
        STRIPE_WEBHOOK_SECRET   — doubles as the HMAC signing secret
        SERVER_BASE_URL         — public base URL of this server
                                  (default: https://super-brain-production.up.railway.app)

Strategy B — R2 presigned URL (if R2_ACCESS_KEY_ID is set):
    Falls back to S3-compatible presigned URL generation.

Email provider (first match wins):
    RESEND_API_KEY          — https://resend.com (3 k free/month)
    SENDGRID_API_KEY        — https://sendgrid.com (100 free/day)
    SMTP_HOST + SMTP_USER + SMTP_PASS  — any SMTP (port 587 TLS)
    (none)                  — logs the URL to stdout; ops can paste manually

Other env vars:
    FROM_EMAIL              — verified sender (default: noreply@example.com)
    SUPPORT_EMAIL           — reply-to in email body
    R2_BUCKET_NAME          — default: super-brain-artifacts
    PDF_OBJECT_KEY          — default: playbook/FULL_PLAYBOOK.pdf
    DOWNLOAD_TTL_SEC        — link expiry in seconds (default: 86400 = 24 h)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via env)
# ---------------------------------------------------------------------------
BUCKET_NAME     = os.environ.get("R2_BUCKET_NAME", "super-brain-artifacts")
PDF_OBJECT_KEY  = os.environ.get("PDF_OBJECT_KEY", "playbook/FULL_PLAYBOOK.pdf")
DOWNLOAD_TTL    = int(os.environ.get("DOWNLOAD_TTL_SEC", "86400"))
PRODUCT_NAME    = "Build Your Own Governed AI Memory Layer - 40-Page Playbook"
EMAIL_SUBJECT   = "Your Playbook - Build Your Own Governed AI Memory Layer"
SERVER_BASE_URL = os.environ.get(
    "SERVER_BASE_URL", "https://super-brain-production.up.railway.app"
).rstrip("/")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def deliver_playbook(to_email: str) -> None:
    """Generate a download URL and email it to *to_email*.

    Swallows exceptions so callers can fire-and-forget.
    """
    try:
        url = _make_download_url(to_email)
        await _send_delivery_email(to_email, url)
        logger.info("delivery: sent playbook to %s", to_email)
    except Exception:
        logger.exception("delivery: failed for %s", to_email)


# ---------------------------------------------------------------------------
# Download URL generation
# ---------------------------------------------------------------------------
def _make_download_url(email: str) -> str:
    """Return a time-limited, HMAC-signed download URL.

    Prefers the server-relay strategy (no R2 S3 creds needed).
    Falls back to R2 presigned URL if R2_ACCESS_KEY_ID is set.
    """
    if os.environ.get("R2_ACCESS_KEY_ID"):
        return _generate_r2_presigned_url()
    return _generate_relay_url(email)


def _generate_relay_url(email: str) -> str:
    """Create a signed token URL that routes through our download proxy."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET is required for relay URL generation")

    expires_at = int(time.time()) + DOWNLOAD_TTL
    payload    = json.dumps({"email": email, "exp": expires_at}, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    token = f"{payload_b64}.{sig}"
    return f"{SERVER_BASE_URL}/api/playbook/download/{token}"


def verify_relay_token(token: str) -> Optional[str]:
    """Verify a relay token. Returns email on success, None on failure."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret or "." not in token:
        return None

    payload_b64, _, sig = token.rpartition(".")
    expected = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None

    # Pad and decode
    padding = "=" * (-len(payload_b64) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except Exception:
        return None

    if int(data.get("exp", 0)) < int(time.time()):
        return None  # expired

    return data.get("email")


async def fetch_pdf_from_r2() -> bytes:
    """Fetch the playbook PDF from R2 via Cloudflare API token."""
    cf_token   = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    account_id = os.environ.get("CF_ACCOUNT_ID", "0adfd16706b3b313d99a9896ec46246c")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/r2/buckets/{BUCKET_NAME}/objects/{PDF_OBJECT_KEY}"
    )
    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {cf_token}"})
        if resp.status_code != 200:
            raise RuntimeError(f"R2 fetch error {resp.status_code}: {resp.text[:200]}")
        return resp.content


# ---------------------------------------------------------------------------
# R2 presigned URL  (AWS S3-compatible SigV4, pure stdlib — Strategy B)
# ---------------------------------------------------------------------------
def _generate_r2_presigned_url() -> str:
    from datetime import datetime, timezone
    from urllib.parse import quote

    access_key_id     = os.environ["R2_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
    account_id        = os.environ.get("CF_ACCOUNT_ID", "0adfd16706b3b313d99a9896ec46246c")

    host     = f"{account_id}.r2.cloudflarestorage.com"
    region, service = "auto", "s3"

    now        = datetime.now(timezone.utc)
    date_stamp = now.strftime("%Y%m%d")
    amz_date   = now.strftime("%Y%m%dT%H%M%SZ")
    cred_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    credential = f"{access_key_id}/{cred_scope}"

    params = sorted([
        ("X-Amz-Algorithm",     "AWS4-HMAC-SHA256"),
        ("X-Amz-Credential",    credential),
        ("X-Amz-Date",          amz_date),
        ("X-Amz-Expires",       str(DOWNLOAD_TTL)),
        ("X-Amz-SignedHeaders", "host"),
    ])
    def _enc(s: str) -> str:
        return quote(s, safe="")
    sorted_query     = "&".join(f"{_enc(k)}={_enc(v)}" for k, v in params)
    canonical_uri    = "/" + BUCKET_NAME + "/" + "/".join(_enc(p) for p in PDF_OBJECT_KEY.split("/"))
    canonical_request = "\n".join([
        "GET", canonical_uri, sorted_query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD",
    ])
    hashed_cr  = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_sig = "\n".join(["AWS4-HMAC-SHA256", amz_date, cred_scope, hashed_cr])

    def _hb(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k = _hb(_hb(_hb(_hb(("AWS4" + secret_access_key).encode(), date_stamp), region), service), "aws4_request")
    sig = hmac.new(k, string_sig.encode(), hashlib.sha256).hexdigest()

    return f"https://{host}/{BUCKET_NAME}/{PDF_OBJECT_KEY}?{sorted_query}&X-Amz-Signature={sig}"


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------
async def _send_delivery_email(to_email: str, download_url: str) -> None:
    from_email    = os.environ.get("FROM_EMAIL", "noreply@example.com")
    support_email = os.environ.get("SUPPORT_EMAIL", "support@example.com")
    html_body     = _build_html(download_url, support_email)
    text_body     = _build_text(download_url, support_email)

    resend_key   = os.environ.get("RESEND_API_KEY")
    sendgrid_key = os.environ.get("SENDGRID_API_KEY")
    smtp_host    = os.environ.get("SMTP_HOST")

    if resend_key:
        await _send_via_resend(resend_key, from_email, to_email, html_body, text_body)
    elif sendgrid_key:
        await _send_via_sendgrid(sendgrid_key, from_email, to_email, html_body, text_body)
    elif smtp_host:
        await asyncio.get_event_loop().run_in_executor(
            None, _send_via_smtp, smtp_host, from_email, to_email, html_body, text_body
        )
    else:
        # Fallback — log the URL so ops can manually deliver
        logger.warning(
            "delivery: no email provider configured. Download URL for %s: %s",
            to_email, download_url,
        )


async def _send_via_resend(api_key: str, from_email: str, to_email: str,
                            html: str, text: str) -> None:
    import httpx
    payload = {
        "from": f"The Super Brain <{from_email}>",
        "to": [to_email],
        "subject": EMAIL_SUBJECT,
        "html": html,
        "text": text,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Resend error {resp.status_code}: {resp.text}")


async def _send_via_sendgrid(api_key: str, from_email: str, to_email: str,
                              html: str, text: str) -> None:
    import httpx
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email, "name": "The Super Brain"},
        "subject": EMAIL_SUBJECT,
        "content": [
            {"type": "text/plain", "value": text},
            {"type": "text/html",  "value": html},
        ],
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.sendgrid.com/v3/mail/send",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"SendGrid error {resp.status_code}: {resp.text}")


def _send_via_smtp(smtp_host: str, from_email: str, to_email: str,
                   html: str, text: str) -> None:
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", from_email)
    smtp_pass = os.environ.get("SMTP_PASS", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"]    = from_email
    msg["To"]      = to_email
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        if smtp_pass:
            smtp.login(smtp_user, smtp_pass)
        smtp.sendmail(from_email, [to_email], msg.as_string())


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------
def _build_html(download_url: str, support_email: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{EMAIL_SUBJECT}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#ffffff;border-radius:8px;overflow:hidden;
                      box-shadow:0 2px 8px rgba(0,0,0,0.08);">
          <tr>
            <td style="background:#1a1a2e;padding:32px 40px;">
              <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">
                The Super Brain
              </h1>
            </td>
          </tr>
          <tr>
            <td style="padding:40px;">
              <h2 style="margin:0 0 16px;color:#1a1a2e;font-size:20px;">
                Your playbook is ready
              </h2>
              <p style="margin:0 0 12px;color:#444;font-size:15px;line-height:1.6;">
                Thank you for your purchase! Your copy of
                <strong>{PRODUCT_NAME}</strong>
                is available via the secure link below.
              </p>
              <p style="margin:0 0 28px;color:#888;font-size:13px;">
                This link expires in <strong>24 hours</strong>.
                Download the PDF now and save it somewhere safe.
              </p>
              <table cellpadding="0" cellspacing="0">
                <tr>
                  <td style="background:#5b50f0;border-radius:6px;">
                    <a href="{download_url}"
                       style="display:inline-block;padding:14px 32px;color:#ffffff;
                              font-size:15px;font-weight:600;text-decoration:none;">
                      Download Your PDF
                    </a>
                  </td>
                </tr>
              </table>
              <p style="margin:32px 0 0;color:#aaa;font-size:12px;line-height:1.6;">
                If the button doesn't work, copy and paste this URL:<br>
                <span style="color:#5b50f0;word-break:break-all;">{download_url}</span>
              </p>
            </td>
          </tr>
          <tr>
            <td style="background:#f9f9f9;padding:24px 40px;border-top:1px solid #eee;">
              <p style="margin:0;color:#aaa;font-size:12px;line-height:1.6;">
                Questions? Contact us at
                <a href="mailto:{support_email}" style="color:#5b50f0;">{support_email}</a>
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _build_text(download_url: str, support_email: str) -> str:
    return f"""Thank you for purchasing "{PRODUCT_NAME}"!

Your secure download link (expires in 24 hours):
{download_url}

Download and save the PDF before the link expires.

---
Questions? Contact us at {support_email}"""
