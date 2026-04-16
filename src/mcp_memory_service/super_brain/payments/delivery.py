"""PDF delivery: R2 presigned URL generation + transactional email.

No new dependencies — uses only stdlib (hashlib, hmac, urllib) and httpx
(already in requirements).

Required env vars:
    R2_ACCESS_KEY_ID        — R2 API token access key
    R2_SECRET_ACCESS_KEY    — R2 API token secret
    CF_ACCOUNT_ID           — Cloudflare account ID
    FROM_EMAIL              — verified sender (e.g. hello@yourdomain.com)
    SUPPORT_EMAIL           — reply-to address shown in email body

Optional:
    R2_BUCKET_NAME          — default: super-brain-artifacts
    PDF_OBJECT_KEY          — default: playbook/FULL_PLAYBOOK.pdf
    PRESIGN_TTL_SEC         — default: 86400 (24 hours)

Email provider (first match wins):
    RESEND_API_KEY          — https://resend.com (3 k free/month)
    SENDGRID_API_KEY        — https://sendgrid.com (100 free/day)
    SMTP_HOST + SMTP_USER + SMTP_PASS  — any SMTP (port 587 TLS)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via env)
# ---------------------------------------------------------------------------
BUCKET_NAME     = os.environ.get("R2_BUCKET_NAME", "super-brain-artifacts")
PDF_OBJECT_KEY  = os.environ.get("PDF_OBJECT_KEY", "playbook/FULL_PLAYBOOK.pdf")
PRESIGN_TTL_SEC = int(os.environ.get("PRESIGN_TTL_SEC", "86400"))
PRODUCT_NAME    = "Build Your Own Governed AI Memory Layer — 40-Page Playbook"
EMAIL_SUBJECT   = "Your Playbook — Build Your Own Governed AI Memory Layer"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
async def deliver_playbook(to_email: str) -> None:
    """Generate a presigned download URL and email it to *to_email*.

    Swallows exceptions so callers can fire-and-forget.
    """
    try:
        url = _generate_presigned_url()
        await _send_delivery_email(to_email, url)
        logger.info("delivery: sent playbook to %s", to_email)
    except Exception:
        logger.exception("delivery: failed for %s", to_email)


# ---------------------------------------------------------------------------
# R2 presigned URL  (AWS S3-compatible SigV4, pure stdlib)
# ---------------------------------------------------------------------------
def _generate_presigned_url() -> str:
    access_key_id     = os.environ["R2_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
    account_id        = os.environ.get("CF_ACCOUNT_ID", "0adfd16706b3b313d99a9896ec46246c")

    host     = f"{account_id}.r2.cloudflarestorage.com"
    region   = "auto"
    service  = "s3"

    now          = datetime.now(timezone.utc)
    date_stamp   = now.strftime("%Y%m%d")
    amz_date     = now.strftime("%Y%m%dT%H%M%SZ")

    cred_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    credential = f"{access_key_id}/{cred_scope}"
    expiry     = str(PRESIGN_TTL_SEC)

    # Canonical query string (lexicographic)
    params = sorted([
        ("X-Amz-Algorithm",     "AWS4-HMAC-SHA256"),
        ("X-Amz-Credential",    credential),
        ("X-Amz-Date",          amz_date),
        ("X-Amz-Expires",       expiry),
        ("X-Amz-SignedHeaders", "host"),
    ])
    sorted_query = "&".join(f"{_rfc3986(k)}={_rfc3986(v)}" for k, v in params)

    canonical_uri     = "/" + BUCKET_NAME + "/" + "/".join(_rfc3986(p) for p in PDF_OBJECT_KEY.split("/"))
    canonical_headers = f"host:{host}\n"
    payload_hash      = "UNSIGNED-PAYLOAD"
    canonical_request = "\n".join([
        "GET", canonical_uri, sorted_query,
        canonical_headers, "host", payload_hash,
    ])

    hashed_canonical = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign   = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, cred_scope, hashed_canonical,
    ])

    signing_key = _derive_signing_key(secret_access_key, date_stamp, region, service)
    signature   = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    endpoint = f"https://{host}/{BUCKET_NAME}/{PDF_OBJECT_KEY}"
    return f"{endpoint}?{sorted_query}&X-Amz-Signature={signature}"


def _derive_signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date    = _hmac_bytes(("AWS4" + secret).encode(), date_stamp)
    k_region  = _hmac_bytes(k_date, region)
    k_service = _hmac_bytes(k_region, service)
    return _hmac_bytes(k_service, "aws4_request")


def _hmac_bytes(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _rfc3986(s: str) -> str:
    return quote(s, safe="")


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
