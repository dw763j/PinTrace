#!/usr/bin/env python3
"""
Email notification helpers for experiment completion.

Usage (import from repository root; optional: ``source .venv/bin/activate``)::

    from scripts.notifier_utils import send_experiment_notification

Environment variables:
    export EXPERIMENT_NOTIFY_ENABLED=1
    export NOTIFY_SMTP_HOST=smtp.example.com
    export NOTIFY_SMTP_PORT=465
    export NOTIFY_SMTP_USERNAME=bot@example.com
    export NOTIFY_SMTP_PASSWORD=your-password
    export NOTIFY_SMTP_USE_TLS=0 # STARTTLS
    export NOTIFY_SMTP_USE_SSL=1 # SMTP over SSL/TLS
    export NOTIFY_FROM=bot@example.com
    export NOTIFY_TO=you@example.com,team@example.com
    export NOTIFY_SUBJECT_PREFIX="[LLM-TPL]"
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a: object, **_kw: object) -> None:
        return None


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _render_summary_rows(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for idx, (key, value) in enumerate(summary.items()):
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = str(value)
        bg_color = "#ffffff" if idx % 2 == 0 else "#f9fafb"
        rows.append(
            f"<tr style='background:{bg_color};'>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;"
            "color:#111827;font-size:13px;font-weight:600;white-space:nowrap;"
            "vertical-align:top;width:30%;'>"
            f"{escape(str(key))}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;"
            "color:#111827;font-size:13px;line-height:1.6;word-break:break-word;"
            "vertical-align:top;'>"
            f"{escape(value_text)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _build_html_email(experiment_name: str, status: str, summary: dict[str, Any]) -> str:
    status_color = "#16a34a" if status.upper() in {"SUCCESS", "NO_JOBS"} else "#dc2626"
    now_utc = datetime.now(timezone.utc).astimezone(timezone(offset=timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")  # display in China Standard Time (UTC+8)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    summary_rows = _render_summary_rows(summary)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(experiment_name)} Notification</title>
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Noto Sans CJK SC','Microsoft YaHei',sans-serif;">
  <div style="max-width:760px;margin:28px auto;padding:0 16px;">
    <div style="background:#111827;color:#fff;border-radius:14px 14px 0 0;padding:20px 24px;">
      <div style="font-size:22px;font-weight:700;line-height:1.2;">Experiment notification</div>
      <div style="margin-top:8px;font-size:14px;color:#d1d5db;">{escape(experiment_name)}</div>
    </div>
    <div style="background:#ffffff;border-radius:0 0 14px 14px;padding:20px 24px;box-shadow:0 8px 24px rgba(0,0,0,0.08);">
      <div style="display:inline-block;padding:6px 12px;border-radius:999px;background:{status_color};color:#fff;font-size:13px;font-weight:700;">
        STATUS: {escape(status)}
      </div>
      <div style="margin-top:10px;font-size:13px;color:#6b7280;">{escape(now_utc)}</div>

      <h3 style="margin:20px 0 10px 0;font-size:16px;color:#111827;">Summary</h3>
      <table style="width:100%;border-collapse:separate;border-spacing:0;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;">
        <thead>
          <tr>
            <th style="text-align:left;background:#eef2ff;color:#3730a3;font-size:13px;padding:11px 12px;border-bottom:1px solid #e5e7eb;font-weight:700;">Field</th>
            <th style="text-align:left;background:#eef2ff;color:#3730a3;font-size:13px;padding:11px 12px;border-bottom:1px solid #e5e7eb;font-weight:700;">Value</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>

      <h3 style="margin:20px 0 10px 0;font-size:16px;color:#111827;">JSON details</h3>
      <pre style="margin:0;background:#0b1020;color:#dbeafe;padding:14px;border-radius:10px;overflow:auto;font-size:12px;line-height:1.5;">{escape(summary_json)}</pre>
    </div>
  </div>
</body>
</html>
"""


def send_experiment_notification(
    *,
    experiment_name: str,
    status: str,
    summary: dict[str, Any],
) -> tuple[bool, str]:
    """
    Send experiment completion email with an HTML summary.

    Returns:
        (sent, message)
    """
    load_dotenv()

    enabled = _truthy(os.getenv("EXPERIMENT_NOTIFY_ENABLED"))
    if not enabled:
        return False, "notification disabled (EXPERIMENT_NOTIFY_ENABLED != true)"

    smtp_host = os.getenv("NOTIFY_SMTP_HOST")
    smtp_port = int(os.getenv("NOTIFY_SMTP_PORT", "587"))
    smtp_user = os.getenv("NOTIFY_SMTP_USERNAME")
    smtp_pass = os.getenv("NOTIFY_SMTP_PASSWORD")
    smtp_use_tls = _truthy(os.getenv("NOTIFY_SMTP_USE_TLS", "1"))
    smtp_use_ssl = _truthy(os.getenv("NOTIFY_SMTP_USE_SSL", "0"))
    mail_from = os.getenv("NOTIFY_FROM") or smtp_user
    mail_to = _split_csv(os.getenv("NOTIFY_TO"))
    subject_prefix = os.getenv("NOTIFY_SUBJECT_PREFIX", "[LLM-TPL]")

    if not smtp_host:
        return False, "missing NOTIFY_SMTP_HOST"
    if not mail_from:
        return False, "missing NOTIFY_FROM (or NOTIFY_SMTP_USERNAME)"
    if not mail_to:
        return False, "missing NOTIFY_TO"
    if smtp_user and not smtp_pass:
        return False, "missing NOTIFY_SMTP_PASSWORD for authenticated SMTP"

    subject = f"{subject_prefix} {experiment_name} - {status}"
    html_body = _build_html_email(experiment_name=experiment_name, status=status, summary=summary)
    text_body = (
        f"{experiment_name} finished with status={status}\n\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        if smtp_use_ssl:
            server: smtplib.SMTP = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        with server:
            if not smtp_use_ssl and smtp_use_tls:
                server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_pass or "")
            server.sendmail(mail_from, mail_to, msg.as_string())
    except Exception as e:
        return False, f"smtp send failed: {type(e).__name__}: {e}"
    return True, "notification sent"

