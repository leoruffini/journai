# message_sender.py defines a class for sending templated messages via WhatsApp using the Twilio API
# The send_templated_message method is used to send a message to a WhatsApp number with a given template from message_templates.py
from twilio.rest import Client #Client is the Twilio API client class that allows us to send messages via WhatsApp
from config import TWILIO_WHATSAPP_NUMBER
import logging
from message_templates import get_message_template #function that returns a message template from message_templates.py

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
            #Create and send a message object using the Twilio API:
            message = self.client.messages.create(  
                body=message_body,
                from_=self.twilio_whatsapp_number,
                to=f'whatsapp:{to_number}'
            )
            self.logger.info(f"Message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            self.logger.error(f"Failed to send message to {to_number}: {str(e)}")