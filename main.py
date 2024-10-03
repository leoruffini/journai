import logging
import openai
from openai import OpenAI
from database import DATABASE_URL
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from twilio.request_validator import RequestValidator
from twilio.rest import Client
import requests
import os
import io
from dotenv import load_dotenv
import datetime
#from google.oauth2.credentials import Credentials
#from googleapiclient.discovery import build
#from google_auth_oauthlib.flow import InstalledAppFlow
#from google.auth.transport.requests import Request as GoogleRequest
from sqlalchemy.orm import Session
from database import get_db, Message, WhitelistedNumber, User
import stripe
from datetime import datetime, timezone, timedelta

load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Google Docs Updater Class
# class GoogleDocsUpdater:
#     def __init__(self, document_id):
#         self.SCOPES = ['https://www.googleapis.com/auth/documents']
#         self.document_id = document_id
#         self.creds = None
#         self.authenticate()

#     def authenticate(self):
#         """Authenticate and create the credentials object."""
#         if os.path.exists('token.json'):
#             self.creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
#         if not self.creds or not self.creds.valid:
#             if self.creds and self.creds.expired and self.creds.refresh_token:
#                 self.creds.refresh(GoogleRequest())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', self.SCOPES)
#                 flow.redirect_uri = 'https://journai.onrender.com/oauth2callback'
#                 self.creds = flow.run_local_server(port=8001)
#             with open('token.json', 'w') as token:
#                 token.write(self.creds.to_json())

#     def update_document(self, transcription_text):
#         """Insert the transcription text into the Google Doc."""
#         try:
#             service = build('docs', 'v1', credentials=self.creds)
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             content = f"\n\nTranscription (at {timestamp}):\n{transcription_text}\n"
#             requests = [{'insertText': {'location': {'index': 1}, 'text': content}}]
#             result = service.documents().batchUpdate(documentId=self.document_id, body={'requests': requests}).execute()
#             logger.info("Document updated successfully.")
#         except Exception as e:
#             logger.error(f"An error occurred while updating the Google Doc: {e}")

# Twilio WhatsApp Handler Class
class TwilioWhatsAppHandler:
    def __init__(self, account_sid: str, auth_token: str, openai_api_key: str):
        # Remove google_docs_updater parameter
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.validator = RequestValidator(auth_token)
        self.openai_api_key = openai_api_key
        self.twilio_client = Client(account_sid, auth_token)
        self.transcription = None
        # Remove self.google_docs_updater = google_docs_updater
        openai.api_key = self.openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)


    def generate_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

    async def generate_response(self, message: str, context: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-3.5-turbo" if GPT-4 is not available
                messages=[
                    {"role": "system", "content": """You are Ada, a concise AI assistant for voice transcription. 
                    Encourage users to send voice messages or subscribe. Keep all responses under 50 words.
                    Do not offer free trials to users who have used all their free trials."""},
                    {"role": "user", "content": f"Context: {context}\nUser message: {message}\nRespond as Ada in under 50 words:"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return "Send a voice message for transcription or subscribe to continue using the service."

    async def handle_whatsapp_request(self, request: Request, db: Session = Depends(get_db)):
        try:
            logger.info("Received WhatsApp request")
            try:
                form_data = await request.form()
            except Exception as e:
                logger.error(f"Error parsing form data: {str(e)}")
                raise HTTPException(status_code=400, detail="Error parsing form data")

            url = str(request.url)
            signature = request.headers.get('X-Twilio-Signature', '')

            logger.info(f"Form data: {form_data}")
            logger.info(f"Request URL: {url}")
            logger.info(f"Signature: {signature}")

            if not self.validator.validate(url, form_data, signature):
                logger.error("Invalid request signature")
                return JSONResponse(content={"message": "Invalid request"}, status_code=400)

            phone_number = form_data.get('From', '').replace('whatsapp:', '')

            user = db.query(User).filter_by(phone_number=phone_number).first()
            if not user:
                # New user, welcome them and enable free trial
                user = User(phone_number=phone_number)
                db.add(user)
                db.commit()
                await self.send_welcome_message(phone_number)
                return JSONResponse(content={"message": "Welcome message sent"}, status_code=200)

            logger.info(f"User found: {user.phone_number}, Free trials remaining: {user.free_trial_remaining}")

            media_type = form_data.get('MediaContentType0')
            if not media_type:
                # Handle text message
                user_message = form_data.get('Body', '')
                
                if user.free_trial_remaining > 0:
                    context = f"User has {user.free_trial_remaining} free trials remaining. Encourage them to use a trial."
                elif self.is_whitelisted(db, phone_number):
                    context = "User is subscribed. Encourage them to use the service."
                else:
                    context = "User has no free trials left and is not subscribed. Acknowledge their message, and gently encourage subscription."
                
                ai_response = await self.generate_response(user_message, context)
                
                if not self.is_whitelisted(db, phone_number) and user.free_trial_remaining == 0:
                    ai_response += "\n\n🔗 Subscribe here: https://buy.stripe.com/test_4gwcMPcx03Et6uk3cc"
                
                await self.send_ai_response(phone_number, ai_response)
                return JSONResponse(content={"message": "Text message handled"}, status_code=200)

            if user.free_trial_remaining > 0:
                user.free_trial_remaining -= 1
                db.commit()
                logger.info(f"User {phone_number} has {user.free_trial_remaining} free trials remaining")
                
                # Process the voice message as usual
                voice_message_url = form_data.get('MediaUrl0')
                logger.info(f"Media type: {media_type}")
                logger.info(f"Voice message URL: {voice_message_url}")

                if not voice_message_url:
                    logger.error("No media found")
                    return JSONResponse(content={"message": "No media found"}, status_code=400)

                if media_type != 'audio/ogg':
                    await self.send_unsupported_media_message(phone_number, media_type)
                    return JSONResponse(content={"message": f"Unsupported media type: {media_type}"}, status_code=400)

                self.transcription = await self.transcribe_voice_message(voice_message_url)
                logger.info(f"Transcription: {self.transcription}")

                embedding = self.generate_embedding(self.transcription)

                db_message = Message(phone_number=phone_number, text=self.transcription, embedding=embedding)
                db.add(db_message)
                db.commit()

                await self.send_transcription(phone_number, self.transcription)

                # If this was the last free trial, send the last free trial message
                if user.free_trial_remaining == 0:
                    await self.send_last_free_trial_message(phone_number)

                return JSONResponse(content={"message": "Voice message processed successfully", "transcription": self.transcription}, status_code=200)
            else:
                logger.info(f"User {phone_number} has no free trials remaining")
                if not self.is_whitelisted(db, phone_number):
                    await self.send_subscription_reminder(phone_number)
                    return JSONResponse(content={"message": "User not subscribed"}, status_code=403)
                else:
                    logger.info(f"User {phone_number} is whitelisted")
                    # Process the voice message as usual for subscribed users
                    voice_message_url = form_data.get('MediaUrl0')
                    logger.info(f"Media type: {media_type}")
                    logger.info(f"Voice message URL: {voice_message_url}")

                    if not voice_message_url:
                        logger.error("No media found")
                        return JSONResponse(content={"message": "No media found"}, status_code=400)

                    if media_type != 'audio/ogg':
                        await self.send_unsupported_media_message(phone_number, media_type)
                        return JSONResponse(content={"message": f"Unsupported media type: {media_type}"}, status_code=400)

                    self.transcription = await self.transcribe_voice_message(voice_message_url)
                    logger.info(f"Transcription: {self.transcription}")

                    embedding = self.generate_embedding(self.transcription)

                    db_message = Message(phone_number=phone_number, text=self.transcription, embedding=embedding)
                    db.add(db_message)
                    db.commit()

                    await self.send_transcription(phone_number, self.transcription)

                    return JSONResponse(content={"message": "Voice message processed successfully", "transcription": self.transcription}, status_code=200)

        except HTTPException as he:
            logger.error(f"HTTP Exception: {str(he)}")
            return JSONResponse(content={"message": str(he)}, status_code=he.status_code)
        except Exception as e:
            logger.exception("Error handling WhatsApp request")
            db.rollback()
            return JSONResponse(content={"message": "Internal server error"}, status_code=500)

    async def generate_response(self, message: str, context: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if GPT-4 is not available
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
            logger.error(f"Error generating AI response: {str(e)}")
            return "I'm here to help with voice transcription. How can I assist you today?"

    async def generate_response(self, message: str, context: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if GPT-4 is not available
                messages=[
                    {"role": "system", "content": """You are Ada, a friendly AI assistant for voice transcription. 
                    Engage in brief, friendly conversation while gently steering users towards using the service or subscribing.
                    Keep responses under 50 words. Be responsive to the user's message, but always relate back to the transcription service.
                    Do not offer free trials to users who have used all their free trials."""},
                    {"role": "user", "content": f"Context: {context}\nUser message: {message}\nRespond as Ada in under 50 words:"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return "I'm here to help with voice transcription. How can I assist you today?"

    async def send_transcription(self, to_number: str, transcription: str):
        try:
            message = self.twilio_client.messages.create(
                body=f"🌟 Your transcript from *Ada*:\n_{transcription}_\n--------------\n🧚‍♂️ Someone fwd this to you? *Try Ada:* https://bit.ly/Free_Ada\u200B",
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Transcription sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send transcription to {to_number}: {str(e)}")

    async def transcribe_voice_message(self, voice_message_url: str) -> str:
        try:
            logger.info(f"Downloading voice message from URL: {voice_message_url}")
            response = requests.get(voice_message_url, auth=(self.account_sid, self.auth_token))
            response.raise_for_status()
            audio_data = response.content
            logger.info("Voice message downloaded successfully")

            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            logger.info("Transcribing voice message using OpenAI")

            audio_file = io.BytesIO(audio_data)
            audio_file.name = "voice_message.ogg"

            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            logger.info("Transcription successful")
            return transcript.text
        except Exception as e:
            logger.exception("Error transcribing voice message")
            return f"Error transcribing voice message: {str(e)}"

    def is_whitelisted(self, db: Session, phone_number: str) -> bool:
        whitelisted = db.query(WhitelistedNumber).filter(
            WhitelistedNumber.phone_number == phone_number,
            WhitelistedNumber.expires_at > datetime.now(timezone.utc)
        ).first()
        return whitelisted is not None

    async def send_subscription_message(self, to_number: str):
        context = "User's free trial has expired and they need to subscribe."
        message = "Subscription needed"
        response = await self.generate_response(message, context)
        response += "\n\nhttps://buy.stripe.com/test_4gwcMPcx03Et6uk3cc"
        await self.send_ai_response(to_number, response)

    async def send_welcome_message(self, to_number: str):
        context = "New user with 3 free trials. Encourage them to try the service immediately."
        message = "Welcome to Ada!"
        response = await self.generate_response(message, context)
        response += "\n\nYou have 3 free transcriptions to try our service. Why not start now? Send me a voice message, and I'll transcribe it for you!"
        await self.send_ai_response(to_number, response)

    async def send_subscription_confirmation(self, to_number: str):
        try:
            message = self.twilio_client.messages.create(
                body="*Welcome to Ada!* 🎉\nThank you for subscribing! You can now enjoy unlimited transcriptions. If you have any questions, feel free to reach out. 😊",
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Subscription confirmation message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send subscription confirmation message to {to_number}: {str(e)}")

    async def send_subscription_cancelled_message(self, to_number: str):
        try:
            message = self.twilio_client.messages.create(
                body="I'm sorry to see you go! 😢 Your subscription has been cancelled. If you change your mind or have any feedback, please don't hesitate to reach out. Thank you for trying me out! 🙏 - Ada",
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Subscription cancelled message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send subscription cancelled message to {to_number}: {str(e)}")

    async def send_last_free_trial_message(self, to_number: str):
        context = "User has just used their last free trial."
        message = "Last free trial used"
        response = await self.generate_response(message, context)
        response += "\n\n🔗 Please subscribe now:\nhttps://buy.stripe.com/test_4gwcMPcx03Et6uk3cc"
        await self.send_ai_response(to_number, response)

    async def send_subscription_reminder(self, to_number: str):
        context = "User needs to subscribe to continue using the service."
        message = "Subscription reminder"
        response = await self.generate_response(message, context)
        response += "\n\n🔗 https://buy.stripe.com/test_4gwcMPcx03Et6uk3cc"
        await self.send_ai_response(to_number, response)

    async def send_unsupported_media_message(self, to_number: str, media_type: str):
        try:
            message = self.twilio_client.messages.create(
                body=(f"Oops! 😅 I can't handle {media_type} files just yet. "
                      "Please send me a voice message, and I'll be happy to transcribe it for you! 🎙️✨\n"
                    ),
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Unsupported media message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send unsupported media message to {to_number}: {str(e)}")

    async def send_text_message_with_free_trials(self, to_number: str, free_trial_remaining: int):
        try:
            message = self.twilio_client.messages.create(
                body=(f"Hi there! 👋 I can only transcribe voice messages. 🎙️✨ "
                      f"You have {free_trial_remaining} free transcriptions left. Please send a voice message to use them."),
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Text message with free trials sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send text message with free trials to {to_number}: {str(e)}")

    async def send_subscribed_user_text_message(self, to_number: str):
        try:
            message = self.twilio_client.messages.create(
                body="Hi there! 👋 I can only transcribe voice messages. 🎙️✨ Please send a voice message to get started.",
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"Subscribed user text message sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send subscribed user text message to {to_number}: {str(e)}")

    async def send_ai_response(self, to_number: str, response: str):
        try:
            message = self.twilio_client.messages.create(
                body=response,
                from_='whatsapp:+12254200006',
                to=f'whatsapp:{to_number}'
            )
            logger.info(f"AI response sent to {to_number}. Message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send AI response to {to_number}: {str(e)}")

# Initialize Google Docs Updater with the Document ID
#google_docs_updater = GoogleDocsUpdater('1LRPBsPdYkgQawy5LxCClyeIrZ-F8T_iJqq2FCQSXNVI')

# Create an instance of TwilioWhatsAppHandler
twilio_whatsapp_handler = TwilioWhatsAppHandler(
    account_sid=os.getenv('TWILIO_ACCOUNT_SID'),
    auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
#    google_docs_updater=google_docs_updater
)

@app.post("/whatsapp", response_model=None)
async def whatsapp(request: Request, db: Session = Depends(get_db)):
    logger.info("Received request to /whatsapp endpoint")
    return await twilio_whatsapp_handler.handle_whatsapp_request(request, db)

@app.get("/transcription", response_model=None)
async def get_transcription(request: Request):
    transcription = twilio_whatsapp_handler.transcription
    return templates.TemplateResponse("transcription.html", {"request": request, "transcription": transcription})

logger.info(f"TWILIO_ACCOUNT_SID: {os.getenv('TWILIO_ACCOUNT_SID')[:5]}...")
logger.info(f"TWILIO_AUTH_TOKEN: {os.getenv('TWILIO_AUTH_TOKEN')[:5]}...")
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...")

stripe.api_key = os.getenv('STRIPE_API_KEY')

@app.post("/create-checkout-session")
async def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price': 'price_1Q4lwXHFdJwdS5kkM8sAAKOJ', 'quantity': 1}],
            mode='subscription',
            success_url='https://journai.onrender.com/success',
            cancel_url='https://journai.onrender.com/cancel',
            phone_number_collection={'enabled': True},
        )
        return RedirectResponse(url=checkout_session.url, status_code=303)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/webhook")
async def webhook_received(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET'))
        logger.info(f"Received Stripe event: {event['type']}")
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            await handle_checkout_completed(session, db)
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            await handle_subscription_deleted(subscription, db)
        else:
            logger.info(f"Unhandled event type: {event['type']}")

        return {"status": "success"}
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail='Invalid payload')
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail='Invalid signature')

async def handle_checkout_completed(session, db: Session):
    logger.info("Processing checkout.session.completed event")
    if session.get('mode') == 'subscription':
        customer_id = session.get('customer')
        if customer_id:
            customer = stripe.Customer.retrieve(customer_id)
            phone_number = customer.get('phone')
            if phone_number:
                # Add or update the whitelist entry
                whitelisted_number = db.query(WhitelistedNumber).filter_by(phone_number=phone_number).first()
                if whitelisted_number:
                    whitelisted_number.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
                else:
                    new_whitelisted_number = WhitelistedNumber(
                        phone_number=phone_number,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=30)
                    )
                    db.add(new_whitelisted_number)
                db.commit()
                logger.info(f"Added or updated whitelist for phone number: {phone_number}")

                # Send subscription confirmation message
                await twilio_whatsapp_handler.send_subscription_confirmation(phone_number)
            else:
                logger.error(f"No phone number found for customer {customer_id}")
        else:
            logger.error("No customer ID found in the checkout session")
    else:
        logger.info("Checkout completed for non-subscription product")

async def handle_subscription_deleted(subscription, db: Session):
    logger.info("Processing customer.subscription.deleted event")
    customer_id = subscription.get('customer')
    if customer_id:
        customer = stripe.Customer.retrieve(customer_id)
        phone_number = customer.get('phone')
        if phone_number:
            db.query(WhitelistedNumber).filter_by(phone_number=phone_number).delete()
            db.commit()
            logger.info(f"Removed {phone_number} from whitelist due to subscription deletion")
            
            # Send the "sorry to see you go" message
            await twilio_whatsapp_handler.send_subscription_cancelled_message(phone_number)
        else:
            logger.error(f"No phone number found for customer {customer_id}")
    else:
        logger.error("No customer ID found in the subscription object")

@app.get("/success")
async def success(request: Request):
    return templates.TemplateResponse("success.html", {"request": request})

@app.get("/cancel")
async def cancel(request: Request):
    return templates.TemplateResponse("cancel.html", {"request": request})

# Retrieve database info
print(f"CONNECTED TO DATABASE: {DATABASE_URL}")