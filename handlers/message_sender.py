from twilio.rest import Client
from config import TWILIO_WHATSAPP_NUMBER
import logging
from message_templates import get_message_template

class MessageSender:
    def __init__(self, account_sid: str, auth_token: str):
        self.client = Client(account_sid, auth_token)
        self.twilio_whatsapp_number = TWILIO_WHATSAPP_NUMBER
        self.logger = logging.getLogger(f"{__name__}.MessageSender")
    
    async def send_templated_message(self, to_number: str, template_key: str, **kwargs):
        try:
            template = get_message_template(template_key)
            if not template:
                self.logger.error(f"Template not found: {template_key}")
                return

            message_body = template.format(**kwargs)
            message = self.client.messages.create(
                body=message_body,
                from_=self.twilio_whatsapp_number,
                to=f'whatsapp:{to_number}'
            )
            self.logger.info(f"Message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            self.logger.error(f"Failed to send message to {to_number}: {str(e)}")