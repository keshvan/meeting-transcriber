from email.message import EmailMessage
from smtplib import SMTP

class SmtpService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        sender: str,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender

    def send_transcript(
        self,
        *,
        recipient: str,
        subject: str,
        html_preview: str,
        txt_content: str,
        attachment_name: str = "transcript.txt",
    ) -> None:
        message = EmailMessage()

        message["From"] = self.sender
        message["To"] = recipient
        message["Subject"] = subject

        message.set_content(
            "Стенограмма встречи готова.\n\n"
            "Полная версия находится во вложении."
        )

        message.add_alternative(html_preview, subtype="html")

        message.add_attachment(
            txt_content.encode("utf-8"),
            maintype="text",
            subtype="plain",
            filename=attachment_name,
        )

        with SMTP(self.host, self.port) as smtp:
            smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.send_message(message)