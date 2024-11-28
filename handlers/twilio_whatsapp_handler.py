import logging
import uuid
from datetime import datetime, timezone

from openai import OpenAI
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from handlers.llm_handler import LLMHandler
from handlers.voice_message_processor import VoiceMessageProcessor
from handlers.stripe_handler import StripeHandler
from handlers.message_sender import MessageSender
from handlers.user_manager import UserManager

from database import User, WhitelistedNumber, Message
from config import (
    BASE_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, OPENAI_API_KEY,
    TWILIO_WHATSAPP_NUMBER, MAX_WHATSAPP_MESSAGE_LENGTH, ADMIN_PHONE_NUMBER
)
from message_templates import get_message_template

class TwilioWhatsAppHandler:
    def __init__(self, db: Session):
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.openai_api_key = OPENAI_API_KEY
        self.twilio_whatsapp_number = TWILIO_WHATSAPP_NUMBER
        self.base_url = BASE_URL
        self.validator = RequestValidator(self.auth_token)
        self.twilio_client = Client(self.account_sid, self.auth_token)
        self.llm_handler = LLMHandler(api_key=self.openai_api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.logger = logging.getLogger(f"{__name__}.TwilioWhatsAppHandler")
        self.stripe_handler = StripeHandler(twilio_handler=self)
        self.voice_message_processor = VoiceMessageProcessor(
            openai_client=self.openai_client,
            llm_handler=self.llm_handler,
            logger=self.logger
        )
        self.message_sender = MessageSender(
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
        self.user_manager = UserManager(db)
        if not all([self.account_sid, self.auth_token, self.openai_api_key, self.twilio_whatsapp_number]):
            raise ValueError("Missing required environment variables for TwilioWhatsAppHandler")

    async def handle_whatsapp_request(self, request: Request, db: Session) -> JSONResponse:
        self.logger.debug("Received WhatsApp request")
        try:
            form_data = await request.form()
            url = str(request.url)
            signature = request.headers.get('X-Twilio-Signature', '')

            self.logger.debug(f"Form data: {form_data}")
            self.logger.debug(f"Request URL: {url}")
            self.logger.debug(f"Signature: {signature}")

            if not self.validator.validate(url, form_data, signature):
                self.logger.warning("Invalid request signature")
                return JSONResponse(content={"message": "Invalid request"}, status_code=400)

            phone_number = form_data.get('From', '').replace('whatsapp:', '')
            user = self.user_manager.get_user_by_phone(phone_number)
            is_subscribed = self.user_manager.is_whitelisted(phone_number)

            media_type = form_data.get('MediaContentType0', '')
            is_voice_message = media_type.startswith('audio/')

            if not user:
                # New user
                user = self.user_manager.create_user(phone_number)

                if is_voice_message:
                    await self.send_templated_message(phone_number, "welcome_with_transcription")
                else:
                    await self.send_templated_message(phone_number, "welcome")

                if not is_voice_message:
                    return JSONResponse(content={"message": "Welcome message sent"}, status_code=200)

            self.logger.info(f"User found: {user.phone_number}, Free trials remaining: {user.free_trial_remaining}, Subscribed: {is_subscribed}")

            if not media_type:
                # Handle text message
                user_message = form_data.get('Body', '')

                if is_subscribed:
                    context = "User is subscribed. Encourage them to use the service."
                elif user.free_trial_remaining > 0:
                    context = f"User has {user.free_trial_remaining} free trials remaining. Encourage them to use a trial."
                else:
                    context = "User has no free trials left and is not subscribed. Acknowledge their message, and gently encourage subscription."

                ai_response = await self.llm_handler.generate_response(user_message, context)

                if not is_subscribed and user.free_trial_remaining == 0:
                    ai_response += f"\n\n🔗 Subscribe here: Please subscribe now:\n{self.stripe_handler.payment_link}"

                await self.send_templated_message(phone_number, "ai_response", response=ai_response)
                return JSONResponse(content={"message": "Text message handled"}, status_code=200)

            if is_subscribed or user.free_trial_remaining > 0:
                if is_voice_message:
                    voice_message_url = form_data.get('MediaUrl0')
                    self.logger.info(f"Media type: {media_type}")
                    self.logger.info(f"Voice message URL: {voice_message_url}")

                    try:
                        transcription = await self.process_voice_message(phone_number, voice_message_url, db)
                        await self.send_admin_notification(phone_number, len(transcription) > MAX_WHATSAPP_MESSAGE_LENGTH, db)
                    except ValueError as e:
                        return JSONResponse(content={"message": str(e)}, status_code=400)

                    if not is_subscribed and user.free_trial_remaining > 0:
                        self.user_manager.update_free_trial(user)
                        self.logger.info(f"User {phone_number} has {user.free_trial_remaining} free trials remaining")

                        if user.free_trial_remaining == 0:
                            await self.send_last_free_trial_message(phone_number)

                    return JSONResponse(content={"message": "Voice message processed successfully", "transcription": transcription}, status_code=200)
                else:
                    await self.send_templated_message(phone_number, "unsupported_media", media_type=media_type)
                    return JSONResponse(content={"message": f"Unsupported media type: {media_type}"}, status_code=400)
            else:
                self.logger.info(f"User {phone_number} has no free trials remaining and is not subscribed")
                await self.send_subscription_reminder(phone_number)
                try:
                    await self.send_admin_notification(phone_number, False, db)
                except Exception as e:
                    self.logger.error(f"Admin notification failed: {str(e)}")
                return JSONResponse(content={"message": "User not subscribed"}, status_code=403)

        except HTTPException as http_exception:
            self.logger.error(f"HTTP Exception: {str(http_exception)}")
            return JSONResponse(content={"message": str(http_exception)}, status_code=http_exception.status_code)
        except Exception as e:
            self.logger.exception("Error handling WhatsApp request")
            db.rollback()
            return JSONResponse(content={"message": "Internal server error"}, status_code=500)

    async def send_admin_notification(self, user_phone: str, summary_generated: bool, db: Session):
        try:
            # Get user status info from db (reuse existing session)
            user = self.user_manager.get_user_by_phone(user_phone)
            whitelisted = self.user_manager.is_whitelisted(user_phone)

            # Build status message
            status_parts = []

            if not user:
                status_parts.append("📥 NEW USER")
            elif isinstance(whitelisted, WhitelistedNumber) or whitelisted is True:
                whitelist_record = db.query(WhitelistedNumber).filter_by(phone_number=user_phone).first()
                if whitelist_record and whitelist_record.expires_at and whitelist_record.expires_at > datetime.now(timezone.utc):
                    status_parts.append(f"💳 PAYING CUSTOMER (expires: {whitelist_record.expires_at.strftime('%Y-%m-%d')})")
                else:
                    status_parts.append("💳 PAYING CUSTOMER (no expiration)")
            elif user.free_trial_remaining > 0:
                status_parts.append(f"🎁 FREE TRIAL USER ({user.free_trial_remaining - 1} remaining)")
            else:
                status_parts.append("🚫 BLOCKED USER (no trials/subscription)")

            message = (
                f"Transcription request from {user_phone}\n"
                f"Status: {' '.join(status_parts)}"
            )

            if summary_generated:
                message += "\nℹ️ Summary was generated"

            self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_whatsapp_number,
                to=ADMIN_PHONE_NUMBER
            )
            self.logger.info(f"Admin notification sent for user {user_phone}")
        except Exception as e:
            self.logger.error(f"Failed to send admin notification: {str(e)}")

    async def send_transcription(self, to_number: str, transcription: str, embedding: list[float], db: Session):
        try:
            # 1. Create message record
            db_message = Message(phone_number=to_number, embedding=embedding)
            db_message.text = transcription

            summary_generated = False

            # 2. Send user messages
            if len(transcription) <= MAX_WHATSAPP_MESSAGE_LENGTH:
                await self.send_templated_message(to_number, "transcription", transcription=transcription)
            else:
                summary_generated = True
                message_hash = uuid.uuid4().hex
                self.logger.info(f"Generated hash for message: {message_hash}")
                db_message.hash = message_hash

                await self.send_templated_message(to_number, "long_transcription_initial", transcription_url=f"{self.base_url}/transcript/{message_hash}")

                summary = await self.llm_handler.generate_summary(transcription)
                await self.send_templated_message(to_number, "long_transcription_summary", summary=summary)

            # 3. Save to database
            db.add(db_message)
            db.commit()

        except Exception as e:
            self.logger.error(f"Failed to send transcription to {to_number}: {str(e)}")
            raise  # Re-raise the exception for main error handling

    async def send_templated_message(self, to_number: str, template_key: str, **kwargs):
        await self.message_sender.send_templated_message(to_number, template_key, **kwargs)

    async def send_last_free_trial_message(self, to_number: str):
        context = "User has just used their last free trial."
        message = "Last free trial used"
        response = await self.llm_handler.generate_response(message, context)
        response += f"\n\n🔗 Please subscribe now:\n{self.stripe_handler.payment_link}"
        await self.send_templated_message(to_number, "subscription_reminder", message=response)

    async def send_subscription_reminder(self, to_number: str):
        context = "User needs to subscribe to continue using the service."
        message = "Subscription reminder"
        response = await self.llm_handler.generate_response(message, context)
        response += f"\n\n🔗 {self.stripe_handler.payment_link}"
        await self.send_templated_message(to_number, "subscription_reminder", message=response)

    async def process_voice_message(self, phone_number: str, voice_message_url: str, db: Session) -> str:
        await self.send_templated_message(phone_number, "processing_confirmation")

        if not voice_message_url:
            self.logger.error("No media found")
            raise ValueError("No media found")

        transcription = await self.voice_message_processor.process_voice_message(
            voice_message_url,
            self.account_sid,
            self.auth_token
        )
        self.logger.info(f"Transcription: {transcription[:50]}")

        # Use the llm_handler instance to generate the embedding
        embedding = self.llm_handler.generate_embedding(transcription)

        await self.send_transcription(phone_number, transcription, embedding, db)

        return transcription