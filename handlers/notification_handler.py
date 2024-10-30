import logging
from twilio.rest import Client
from config import TWILIO_WHATSAPP_NUMBER, ADMIN_PHONE_NUMBER

class NotificationHandler:
    def __init__(self, account_sid: str, auth_token: str):
        self.twilio_client = Client(account_sid, auth_token)
        self.from_number = TWILIO_WHATSAPP_NUMBER
        self.admin_number = ADMIN_PHONE_NUMBER
        self.logger = logging.getLogger(__name__)

    async def send_admin_notification(self, message: str):
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.admin_number
            )
            self.logger.info("Admin notification sent")
        except Exception as e:
            self.logger.error(f"Failed to send admin notification: {str(e)}")