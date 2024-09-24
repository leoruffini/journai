import logging
import openai
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from twilio.request_validator import RequestValidator
from twilio.rest import Client
import requests
import os
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

class TwilioWhatsAppHandler:

    def __init__(self, account_sid: str, auth_token: str, openai_api_key: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.validator = RequestValidator(auth_token)
        self.openai_api_key = openai_api_key
        self.twilio_client = Client(account_sid, auth_token)
        self.transcription = None  # Initialize transcription attribute

    async def handle_whatsapp_request(self, request: Request):
        try:
            logger.info("Received WhatsApp request")
            # Validate the request
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

            # Get the voice message URL
            voice_message_url = form_data.get('MediaUrl0')
            logger.info(f"Voice message URL: {voice_message_url}")

            if not voice_message_url:
                logger.error("No voice message found")
                return JSONResponse(content={"message": "No voice message found"}, status_code=400)

            # Transcribe the voice message
            self.transcription = await self.transcribe_voice_message(voice_message_url)

            # Log the transcription
            logger.info(f"Transcription: {self.transcription}")

            # Return a success message
            return JSONResponse(content={"message": "Voice message processed successfully", "transcription": self.transcription}, status_code=200)

        except HTTPException as he:
            logger.error(f"HTTP Exception: {str(he)}")
            return JSONResponse(content={"message": str(he)}, status_code=he.status_code)
        except Exception as e:
            logger.exception("Error handling WhatsApp request")
            return JSONResponse(content={"message": "Internal server error"}, status_code=500)

    async def transcribe_voice_message(self, voice_message_url: str) -> str:
        try:
            logger.info(f"Downloading voice message from URL: {voice_message_url}")

            # Download the media content directly
            response = requests.get(voice_message_url, auth=(self.account_sid, self.auth_token))
            response.raise_for_status()

            audio_data = response.content
            logger.info("Voice message downloaded successfully")

            # Use OpenAI's new client to transcribe the audio
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            logger.info("Transcribing voice message using OpenAI")

            # Create a file-like object from the audio data
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "voice_message.ogg"  # Adjust the file extension as needed

            # Transcribe the audio file using the new method
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            logger.info("Transcription successful")
            return transcript.text
        except Exception as e:
            logger.exception("Error transcribing voice message")
            return f"Error transcribing voice message: {str(e)}"

# Create an instance of TwilioWhatsAppHandler
twilio_whatsapp_handler = TwilioWhatsAppHandler(
    account_sid=os.getenv('TWILIO_ACCOUNT_SID'),
    auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

@app.post("/whatsapp")
async def whatsapp(request: Request):
    logger.info("Received request to /whatsapp endpoint")
    logger.info(f"Request headers: {request.headers}")
    body = await request.body()
    logger.info(f"Request body: {body}")
    return await twilio_whatsapp_handler.handle_whatsapp_request(request)

@app.get("/transcription")
async def get_transcription(request: Request):
    transcription = twilio_whatsapp_handler.transcription
    return templates.TemplateResponse("transcription.html", {"request": request, "transcription": transcription})

# Log environment variables to ensure they're loaded correctly (masking sensitive data)
logger.info(f"TWILIO_ACCOUNT_SID: {os.getenv('TWILIO_ACCOUNT_SID')[:5]}...")
logger.info(f"TWILIO_AUTH_TOKEN: {os.getenv('TWILIO_AUTH_TOKEN')[:5]}...")
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...")

# Log OpenAI library version and module path
logger.info(f"OpenAI Library Version: {openai.__version__}")
logger.info(f"OpenAI Module File: {openai.__file__}")


