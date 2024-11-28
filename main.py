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
from handlers.llm_handler import LLMHandler
from handlers.voice_message_processor import VoiceMessageProcessor
from handlers.stripe_handler import StripeHandler
from handlers.twilio_whatsapp_handler import TwilioWhatsAppHandler
from database import DATABASE_URL, Message, WhitelistedNumber, User, get_db
from config import (
    BASE_URL, STRIPE_PAYMENT_LINK, STRIPE_CUSTOMER_PORTAL_URL,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, OPENAI_API_KEY,
    TWILIO_WHATSAPP_NUMBER, LOG_LEVEL, MAX_WHATSAPP_MESSAGE_LENGTH,
    STRIPE_WEBHOOK_SECRET, STRIPE_API_KEY,
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

# Create an instance of TwilioWhatsAppHandler with dependency injection
@app.post("/whatsapp", response_model=None)
async def whatsapp(request: Request, db: Session = Depends(get_db)):
    twilio_whatsapp_handler = TwilioWhatsAppHandler(db)
    logger.debug("Received request to /whatsapp endpoint")
    return await twilio_whatsapp_handler.handle_whatsapp_request(request, db)

logger.info(f"TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID[:8]}...")
logger.info(f"TWILIO_AUTH_TOKEN: {TWILIO_AUTH_TOKEN[:8]}...")
logger.info(f"OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...")
logger.info(f"STRIPE_WEBHOOK_SECRET: {STRIPE_WEBHOOK_SECRET[:16]}...")
logger.info(f"DEBUG MODE: {LOG_LEVEL}")

# Initialize StripeHandler with TwilioWhatsAppHandler
twilio_whatsapp_handler = TwilioWhatsAppHandler(get_db())
stripe_handler = StripeHandler(twilio_handler=twilio_whatsapp_handler)

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