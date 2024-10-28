# Standard library imports
import io
import logging
import uuid
from datetime import datetime, timezone

# Third-party imports
from openai import OpenAI
import requests
import stripe
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from twilio.request_validator import RequestValidator
from twilio.rest import Client

# Local imports
from database import DATABASE_URL, Message, WhitelistedNumber, User, get_db
from config import (
    BASE_URL, STRIPE_PAYMENT_LINK, STRIPE_CUSTOMER_PORTAL_URL,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, OPENAI_API_KEY,
    TWILIO_WHATSAPP_NUMBER, LOG_LEVEL, MAX_WHATSAPP_MESSAGE_LENGTH,
    TRANSCRIPTION_MODEL, LLM_MODEL, STRIPE_WEBHOOK_SECRET, STRIPE_API_KEY,
    ADMIN_PHONE_NUMBER
)
from message_templates import get_message_template

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

class LLMHandler:
    def __init__(self, api_key: str, model: str = LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger(f"{__name__}.LLMHandler")

    async def generate_response(self, message: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are Ada, a friendly AI assistant for voice transcription. 
                    Engage in brief, friendly conversation while gently steering users towards using the service or subscribing.
                    Keep responses under 50 words. Be responsive to the user's message, but always relate back to the transcription service.
                    Do not offer free trials or transcription services to users who have used all their free trials and are not subscribed.
                    For users with free trials, encourage them to use the service.
                    For subscribed users, remind them of the benefits and encourage use.
                    For users without free trials or subscription, focus on the benefits of subscribing."""},
                    {"role": "user", "content": f"Context: {context}\nUser message: {message}\nRespond as Ada in under 50 words:"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating AI response: {str(e)}")
            return "I'm here to help with voice transcription. How can I assist you today?"

    async def generate_summary(self, transcription: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are Ada, a top-tier AI assistant specializing in summarizing voice message transcriptions. Your summaries must **absolutely not exceed 1500 characters**. Before finalizing the summary, **internally count the characters** to ensure compliance. It's essential to **preserve the original tone, conversational style, and personal touches of the speaker**, including any colloquial expressions, rhetorical questions, and informal language. Maintain the original language variant and regional differences (e.g., Spain Spanish vs. Latin American Spanish) when summarizing. Do not mention the character count in your final summary."""},
                    {"role": "user", "content": f"""Please summarize the following transcription. Ensure the summary is concise and complete, capturing the key points succinctly. It is crucial that the summary **does not exceed 1500 characters**. **Internally verify the character count** before providing the final summary. Focus on key points while **maintaining the speaker's tone, conversational style, and personal expressions**. Preserve any colloquial phrases, rhetorical questions, and regional language variants, including any regional Spanish differences:\n\n{transcription}"""}
                ],
                max_tokens=300  # Adjust as needed to ensure the summary stays under 1500 characters
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "I apologize, but I encountered an error while summarizing your message. Please try again later."


class StripeHandler:
    def __init__(self):
        self.api_key = STRIPE_API_KEY
        self.webhook_secret = STRIPE_WEBHOOK_SECRET
        self.payment_link = STRIPE_PAYMENT_LINK
        self.customer_portal_url = STRIPE_CUSTOMER_PORTAL_URL
        stripe.api_key = self.api_key

        if not all([self.api_key, self.webhook_secret, self.payment_link, self.customer_portal_url]):
            raise ValueError("Missing required environment variables for StripeHandler")

    def create_checkout_session(self):
        return RedirectResponse(url=self.payment_link, status_code=303)

    def construct_event(self, payload, sig_header):
        return stripe.Webhook.construct_event(payload, sig_header, self.webhook_secret)

    async def handle_checkout_completed(self, session, db: Session):
        logger.info("Processing checkout.session.completed event")
        if session.get('mode') == 'subscription':
            customer_id = session.get('customer')
            if customer_id:
                customer = stripe.Customer.retrieve(customer_id)
                phone_number = customer.get('phone')
                if phone_number:
                    subscription_id = session.get('subscription')
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    current_period_end = datetime.fromtimestamp(subscription.current_period_end, timezone.utc)
                    whitelisted_number = db.query(WhitelistedNumber).filter_by(phone_number=phone_number).first()
                    if not whitelisted_number:
                        whitelisted_number = WhitelistedNumber(phone_number=phone_number)
                        db.add(whitelisted_number)
                    whitelisted_number.expires_at = current_period_end
                    db.commit()
                    logger.info(f"Added or updated whitelist for phone number: {phone_number}, expires at: {current_period_end}")
                    await twilio_whatsapp_handler.send_subscription_confirmation(phone_number)
                else:
                    logger.error(f"No phone number found for customer {customer_id}")
            else:
                logger.error("No customer ID found in the checkout session")
        else:
            logger.info("Checkout completed for non-subscription product")

    async def handle_subscription_deleted(self, subscription, db: Session):
        logger.info("Processing customer.subscription.deleted event")
        customer_id = subscription.get('customer')
        if customer_id:
            customer = stripe.Customer.retrieve(customer_id)
            phone_number = customer.get('phone')
            if phone_number:
                db.query(WhitelistedNumber).filter_by(phone_number=phone_number).delete()
                db.commit()
                logger.info(f"Removed {phone_number} from whitelist due to subscription deletion")
                await twilio_whatsapp_handler.send_subscription_cancelled_message(phone_number)
            else:
                logger.error(f"No phone number found for customer {customer_id}")
        else:
            logger.error("No customer ID found in the subscription object")

    async def handle_subscription_updated(self, subscription, db: Session):
        customer_id = subscription.get('customer')
        if customer_id:
            customer = stripe.Customer.retrieve(customer_id)
            phone_number = customer.get('phone')
            if phone_number:
                current_period_end = datetime.fromtimestamp(subscription.current_period_end, timezone.utc)
                whitelisted_number = db.query(WhitelistedNumber).filter_by(phone_number=phone_number).first()
                if whitelisted_number:
                    whitelisted_number.expires_at = current_period_end
                    db.commit()
                    logger.info(f"Updated expiration for {phone_number} to {current_period_end}")
                else:
                    logger.error(f"Whitelisted number not found for phone: {phone_number}")
            else:
                logger.error(f"No phone number found for customer {customer_id}")
        else:
            logger.error("No customer ID found in the subscription object")

class VoiceMessageProcessor:
    def __init__(self, openai_client: OpenAI, llm_handler: LLMHandler, logger: logging.Logger):
        self.openai_client = openai_client
        self.llm_handler = llm_handler
        self.logger = logger

    async def process_voice_message(self, voice_message_url: str, account_sid: str, auth_token: str) -> str:
        try:
            audio_data = await self.download_voice_message(voice_message_url, account_sid, auth_token)
            raw_transcription = await self.transcribe_voice_message(audio_data)
            post_processed_transcript = await self.post_process_transcription(raw_transcription)
            return post_processed_transcript
        except Exception as e:
            self.logger.exception("Error processing voice message")
            return f"Error processing voice message: {str(e)}"

    async def download_voice_message(self, voice_message_url: str, account_sid: str, auth_token: str) -> bytes:
        self.logger.info(f"Downloading voice message from URL: {voice_message_url}")
        response = requests.get(voice_message_url, auth=(account_sid, auth_token))
        response.raise_for_status()
        self.logger.info("Voice message downloaded successfully")
        return response.content

    async def transcribe_voice_message(self, audio_data: bytes) -> str:
        self.logger.info("Transcribing voice message using OpenAI")
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "voice_message.ogg"
        transcript = self.openai_client.audio.transcriptions.create(model=TRANSCRIPTION_MODEL, file=audio_file)
        self.logger.info("Transcription successful")
        return transcript.text

    async def post_process_transcription(self, transcription: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You are an expert multilingual transcription assistant. Your task is to **post-process transcriptions of voice messages** in different languages, including but not limited to Spanish, Catalan, and English, to enhance their readability and accuracy **before they are summarized**. Specifically, you should:

1. **Detect the language of the transcription** (most probably Spanish, Catalan, or English) and apply language-specific rules for spelling, grammar, and punctuation. Correct any errors while preserving regional language variants (e.g., Spain Spanish vs. Latin American Spanish, Catalan, or English dialects).
2. **Add appropriate punctuation and capitalization** to clarify meaning and improve readability, without altering the speaker's intended message.
3. **Preserve the original tone, style, and personal expressions** of the speaker, including colloquial phrases and regionalisms, based on the detected language.
4. **Do not add, remove, or alter any content beyond what is necessary for correction**. Keep the text as close to the original meaning as possible.

Provide the corrected transcription only, without any additional comments or explanations."""},
                    {"role": "user", "content": f"""Please post-process the following transcription of a voice message. The transcription may be in Spanish, Catalan, or English. 

**Instructions:**

- Detect the language and apply language-specific corrections for spelling, grammar, and punctuation.
- Correct any errors and add necessary punctuation and capitalization.
- Preserve the speaker's tone, style, and regional language variants.
- Do not change the original meaning or omit any parts of the text.

Provide the corrected transcription only:\n\n{transcription}"""}
                ],
                max_tokens=1500  # Adjust as needed
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error post-processing transcription: {str(e)}")
            return transcription  # Return the original transcription if post-processing fails

# Twilio WhatsApp Handler Class
class TwilioWhatsAppHandler:
    def __init__(self):
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
        self.stripe_handler = StripeHandler()
        self.voice_message_processor = VoiceMessageProcessor(
            openai_client=self.openai_client,
            llm_handler=self.llm_handler,
            logger=self.logger
        )

        if not all([self.account_sid, self.auth_token, self.openai_api_key, self.twilio_whatsapp_number]):
            raise ValueError("Missing required environment variables for TwilioWhatsAppHandler")

    def generate_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

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
            user = db.query(User).filter_by(phone_number=phone_number).first()
            is_subscribed = self.is_whitelisted(db, phone_number)

            media_type = form_data.get('MediaContentType0', '')
            is_voice_message = media_type.startswith('audio/')

            if not user:
                # New user
                user = User(phone_number=phone_number)
                db.add(user)
                db.commit()
                
                if is_voice_message:
                    await self.send_welcome_with_transcription_info(phone_number)
                else:
                    await self.send_welcome_message(phone_number)
                
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
                
                await self.send_ai_response(phone_number, ai_response)
                return JSONResponse(content={"message": "Text message handled"}, status_code=200)

            if is_subscribed or user.free_trial_remaining > 0:
                if is_voice_message:
                    voice_message_url = form_data.get('MediaUrl0')
                    self.logger.info(f"Media type: {media_type}")
                    self.logger.info(f"Voice message URL: {voice_message_url}")

                    try:
                        transcription = await self.process_voice_message(phone_number, voice_message_url, db)
                    except ValueError as e:
                        return JSONResponse(content={"message": str(e)}, status_code=400)

                    if not is_subscribed and user.free_trial_remaining > 0:
                        user.free_trial_remaining -= 1
                        db.commit()
                        self.logger.info(f"User {phone_number} has {user.free_trial_remaining} free trials remaining")

                        if user.free_trial_remaining == 0:
                            await self.send_last_free_trial_message(phone_number)

                    return JSONResponse(content={"message": "Voice message processed successfully", "transcription": transcription}, status_code=200)
                else:
                    await self.send_unsupported_media_message(phone_number, media_type)
                    return JSONResponse(content={"message": f"Unsupported media type: {media_type}"}, status_code=400)
            else:
                self.logger.info(f"User {phone_number} has no free trials remaining and is not subscribed")
                await self.send_subscription_reminder(phone_number)
                return JSONResponse(content={"message": "User not subscribed"}, status_code=403)

        except HTTPException as http_exception:
            self.logger.error(f"HTTP Exception: {str(http_exception)}")
            return JSONResponse(content={"message": str(http_exception)}, status_code=http_exception.status_code)
        except Exception as e:
            self.logger.exception("Error handling WhatsApp request")
            db.rollback()
            return JSONResponse(content={"message": "Internal server error"}, status_code=500)

    async def send_admin_notification(self, user_phone: str, summary_generated: bool):
        try:
            message = f"New transcription request from {user_phone}."
            if summary_generated:
                message += " A summary was generated."
            
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
            db_message = Message(phone_number=to_number, embedding=embedding)
            db_message.text = transcription

            summary_generated = False
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

            db.add(db_message)
            db.commit()

            # Send admin notification
            await self.send_admin_notification(to_number, summary_generated)

        except Exception as e:
            self.logger.error(f"Failed to send transcription to {to_number}: {str(e)}")

    async def send_templated_message(self, to_number: str, template_key: str, **kwargs):
        try:
            template = get_message_template(template_key)
            if not template:
                self.logger.error(f"Template not found: {template_key}")
                return

            message_body = template.format(**kwargs)
            message = self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_whatsapp_number,
                to=f'whatsapp:{to_number}'
            )
            self.logger.info(f"Message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            self.logger.error(f"Failed to send message to {to_number}: {str(e)}")

    async def send_welcome_message(self, to_number: str):
        await self.send_templated_message(to_number, "welcome")

    async def send_welcome_with_transcription_info(self, to_number: str):
        await self.send_templated_message(to_number, "welcome_with_transcription")

    async def send_processing_confirmation(self, to_number: str):
        await self.send_templated_message(to_number, "processing_confirmation")

    async def send_subscription_cancelled_message(self, to_number: str):
        await self.send_templated_message(to_number, "subscription_cancelled")

    async def send_unsupported_media_message(self, to_number: str, media_type: str):
        await self.send_templated_message(to_number, "unsupported_media", media_type=media_type)

    async def send_text_message_with_free_trials(self, to_number: str, free_trial_remaining: int):
        await self.send_templated_message(to_number, "text_message_with_free_trials", free_trial_remaining=free_trial_remaining)

    async def send_subscribed_user_text_message(self, to_number: str):
        await self.send_templated_message(to_number, "subscribed_user_text_message")

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

    async def send_ai_response(self, to_number: str, response: str):
        await self.send_templated_message(to_number, "ai_response", response=response)

    async def send_subscription_confirmation(self, to_number: str):
        context = "User has successfully subscribed to the service."
        message = "Subscription confirmation"
        response = await self.llm_handler.generate_response(message, context)
        response += f"\n\n```MANAGE YOUR SUBSCRIPTION: {self.stripe_handler.customer_portal_url}```"
        await self.send_templated_message(to_number, "subscription_confirmation", response=response)

    async def process_voice_message(self, phone_number: str, voice_message_url: str, db: Session) -> str:
        await self.send_processing_confirmation(phone_number)

        if not voice_message_url:
            self.logger.error("No media found")
            raise ValueError("No media found")

        transcription = await self.voice_message_processor.process_voice_message(
            voice_message_url,
            self.account_sid,
            self.auth_token
        )
        self.logger.info(f"Transcription: {transcription[:50]}")

        embedding = self.generate_embedding(transcription)

        await self.send_transcription(phone_number, transcription, embedding, db)

        return transcription

    def is_whitelisted(self, db: Session, phone_number: str) -> bool:
        whitelisted = db.query(WhitelistedNumber).filter(
            WhitelistedNumber.phone_number == phone_number,
            WhitelistedNumber.expires_at > datetime.now(timezone.utc)
        ).first()
        return whitelisted is not None

# Create an instance of TwilioWhatsAppHandler
twilio_whatsapp_handler = TwilioWhatsAppHandler()

@app.post("/whatsapp", response_model=None)
async def whatsapp(request: Request, db: Session = Depends(get_db)):
    logger.debug("Received request to /whatsapp endpoint")
    return await twilio_whatsapp_handler.handle_whatsapp_request(request, db)

logger.info(f"TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID[:8]}...")
logger.info(f"TWILIO_AUTH_TOKEN: {TWILIO_AUTH_TOKEN[:8]}...")
logger.info(f"OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...")
logger.info(f"STRIPE_WEBHOOK_SECRET: {STRIPE_WEBHOOK_SECRET[:16]}...")
logger.info(f"DEBUG MODE: {LOG_LEVEL}")

# Create an instance of StripeHandler
stripe_handler = StripeHandler()

@app.post("/create-checkout-session")
async def create_checkout_session():
    return stripe_handler.create_checkout_session()

@app.post("/webhook")
async def webhook_received(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe_handler.construct_event(payload, sig_header)
        logger.info(f"Received Stripe event: {event['type']}")

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            await stripe_handler.handle_checkout_completed(session, db)
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            await stripe_handler.handle_subscription_deleted(subscription, db)
        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            await stripe_handler.handle_subscription_updated(subscription, db)
        else:
            logger.info(f"Unhandled event type: {event['type']}")

        return {"status": "success"}
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail='Invalid payload')
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail='Invalid signature')
    except stripe.error.AuthenticationError as e:
        logger.error(f"Stripe authentication error: {e}")
        raise HTTPException(status_code=500, detail='Stripe authentication error')
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail='Stripe error')
    except Exception as e:
        logger.error(f"Unexpected error in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail='Internal server error')

@app.get("/success")
async def success(request: Request):
    return templates.TemplateResponse("success.html", {"request": request})

@app.get("/cancel")
async def cancel(request: Request):
    return templates.TemplateResponse("cancel.html", {"request": request})

@app.get("/transcript/{message_hash}", response_model=None)
async def get_transcription_by_hash(message_hash: str, request: Request, db: Session = Depends(get_db)):
    logger.info(f"Attempting to retrieve message with hash: {message_hash}")

    # Get the User-Agent header
    user_agent = request.headers.get('User-Agent', '')
    logger.info(f"User-Agent: {user_agent}")

    # Define a list of known pre-fetcher User-Agents
    prefetch_user_agents = [
        'WhatsApp',
        'facebookexternalhit',
        'Facebot',
        # Add other known pre-fetcher identifiers if necessary
    ]

    # Check if the User-Agent belongs to a pre-fetcher
    if any(agent in user_agent for agent in prefetch_user_agents):
        logger.info("Detected pre-fetch request. Serving minimal response.")
        # Serve a minimal response without deleting the hash
        return Response(status_code=200)

    # Proceed with normal logic for actual user requests
    try:
        # Query the database for the message with the given hash
        db_message = db.query(Message).filter(Message.hash == message_hash).first()

        if not db_message:
            logger.error(f"Transcription not found for hash: {message_hash}")
            return templates.TemplateResponse("transcript.html", {
                "request": request,
                "transcription": None,
                "error_message": "🚨 Error: Transcription not found or already viewed"
            })

        logger.info(f"Found message. First 100 characters: {db_message.text[:100]}...")

        # Return the transcription
        response = templates.TemplateResponse("transcript.html", {
            "request": request,
            "transcription": db_message.text,  # This will use the getter to decrypt
            "error_message": None
        })

        # Delete the hash to prevent future access
        db_message.hash = None
        db.commit()

        return response

    except Exception as e:
        logger.error(f"Error retrieving transcription: {str(e)}")
        return templates.TemplateResponse("transcript.html", {
            "request": request,
            "transcription": None,
            "error_message": "An error occurred while retrieving the transcription"
        })

# Retrieve database info
print(f"CONNECTED TO DATABASE: {DATABASE_URL}")