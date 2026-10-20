import os
import smtplib
from email.message import EmailMessage


DEFAULT_SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
DEFAULT_SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))


def _build_dispatch_message(alert):
    timestamp_text = alert.dispatched_at.isoformat() if alert.dispatched_at else "N/A"
    confidence_text = (
        f"{alert.prediction_confidence:.2f}%"
        if alert.prediction_confidence is not None
        else "N/A"
    )

    lines = [
        f"TARS dispatch alert for MMSI {alert.ship_id}",
        "",
        f"Status: {alert.status}",
        f"Coordinates: {alert.lat}, {alert.lon}",
        f"Confidence: {confidence_text}",
        f"Dispatch timestamp: {timestamp_text}",
        f"Anomaly timestamp: {alert.anomaly_timestamp.isoformat() if alert.anomaly_timestamp else 'N/A'}",
    ]

    if alert.image_url:
        lines.append(f"Image link: {alert.image_url}")

    return "\n".join(lines)


def send_dispatch_notification(alert):
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    alert_recipient = os.getenv("ALERT_RECIPIENT")
    message_body = _build_dispatch_message(alert)

    if not all([email_user, email_pass, alert_recipient]):
        fallback_message = (
            f"[SIMULATED EMAIL SENT] Alert dispatched for MMSI {alert.ship_id}\n"
            f"{message_body}"
        )
        print(fallback_message, flush=True)
        return False, fallback_message

    message = EmailMessage()
    message["Subject"] = f"TARS Dispatch Alert: MMSI {alert.ship_id}"
    message["From"] = email_user
    message["To"] = alert_recipient
    message.set_content(message_body)

    try:
        with smtplib.SMTP(DEFAULT_SMTP_HOST, DEFAULT_SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(message)

        success_message = f"Dispatch email sent to {alert_recipient}"
        print(success_message, flush=True)
        return True, success_message
    except Exception as exc:
        fallback_message = (
            f"[SIMULATED EMAIL SENT] Alert dispatched for MMSI {alert.ship_id} "
            f"(email send failed: {exc})\n{message_body}"
        )
        print(fallback_message, flush=True)
        return False, fallback_message
