import logging
import openai
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from twilio.request_validator import RequestValidator
from twilio.rest import Client
import requests
import os
import io
from dotenv import load_dotenv
import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleRequest
from sqlalchemy.orm import Session
from database import get_db, Message

load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Google Docs Updater Class
class GoogleDocsUpdater:

    def __init__(self, document_id):
        self.SCOPES = ['https://www.googleapis.com/auth/documents']
        self.document_id = document_id
        self.creds = None
        self.authenticate()

    def authenticate(self):
        """Authenticate and create the credentials object."""
        # Check if there are already stored credentials
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)

        # If there are no valid credentials, authenticate the user
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(GoogleRequest())  # Use Google OAuth's Request
            else:
                # Set up a static redirect URI and force the OAuth flow to use it
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secret.json', self.SCOPES)
                
                # Explicitly set the redirect_uri to match the one registered in Google Cloud
                flow.redirect_uri = 'https://journai.onrender.com/oauth2callback'
                
                # Run the local server without passing redirect_uri again
                self.creds = flow.run_local_server(port=8001)

            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())

    def update_document(self, transcription_text):
        """Insert the transcription text into the Google Doc."""
        try:
            # Create a service object to interact with the Docs API
            service = build('docs', 'v1', credentials=self.creds)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Define the content to insert
            content = f"\n\nTranscription (at {timestamp}):\n{transcription_text}\n"

            # Create the request body for the batchUpdate
            requests = [
                {
                    'insertText': {
                        'location': {
                            'index': 1,  # Insert after the first index (beginning of the doc)
                        },
                        'text': content
                    }
                }
            ]

            # Send the request to update the document
            result = service.documents().batchUpdate(
                documentId=self.document_id,
                body={'requests': requests}
            ).execute()

            logger.info("Document updated successfully.")
        except Exception as e:
            logger.error(f"An error occurred while updating the Google Doc: {e}")

# Twilio WhatsApp Handler Class
class TwilioWhatsAppHandler:

    def __init__(self, account_sid: str, auth_token: str, openai_api_key: str, google_docs_updater: GoogleDocsUpdater):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.validator = RequestValidator(auth_token)
        self.openai_api_key = openai_api_key
        self.twilio_client = Client(account_sid, auth_token)
        self.transcription = None  # Initialize transcription attribute
        self.google_docs_updater = google_docs_updater  # Add the Google Docs updater instance
        openai.api_key = self.openai_api_key  # Set OpenAI API key
        self.openai_client = OpenAI(api_key=openai_api_key)  # Create OpenAI client

    def generate_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    async def handle_whatsapp_request(self, request: Request, db: Session = Depends(get_db)):
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

            # Get the phone number from the form data and remove the "whatsapp:" prefix
            phone_number = form_data.get('From', '')
            phone_number = phone_number.replace('whatsapp:', '')
            

            # Generate embedding for the transcription
            embedding = self.generate_embedding(self.transcription)

            # Store message in database
            db_message = Message(
                phone_number=phone_number,  # This will now be without the "whatsapp:" prefix
                text=self.transcription,
                embedding=embedding
            )
            db.add(db_message)
            db.commit()

            # Update Google Docs with the transcription and timestamp
            self.google_docs_updater.update_document(self.transcription)

            # Return a success message
            return JSONResponse(content={"message": "Voice message processed successfully", "transcription": self.transcription}, status_code=200)

        except HTTPException as he:
            logger.error(f"HTTP Exception: {str(he)}")
            return JSONResponse(content={"message": str(he)}, status_code=he.status_code)
        except Exception as e:
            logger.exception("Error handling WhatsApp request")
            db.rollback()  # Roll back the transaction in case of error
            return JSONResponse(content={"message": "Internal server error"}, status_code=500)

    async def transcribe_voice_message(self, voice_message_url: str) -> str:
        try:
            logger.info(f"Downloading voice message from URL: {voice_message_url}")

            # Download the media content directly using Twilio credentials
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

# Initialize Google Docs Updater with the Document ID
google_docs_updater = GoogleDocsUpdater('1LRPBsPdYkgQawy5LxCClyeIrZ-F8T_iJqq2FCQSXNVI')

# Create an instance of TwilioWhatsAppHandler
twilio_whatsapp_handler = TwilioWhatsAppHandler(
    account_sid=os.getenv('TWILIO_ACCOUNT_SID'),
    auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    google_docs_updater=google_docs_updater  # Pass the Google Docs updater to the handler
)

@app.post("/whatsapp", response_model=None)  # Disable response model validation
async def whatsapp(request: Request, db: Session = Depends(get_db)):
    logger.info("Received request to /whatsapp endpoint")
    return await twilio_whatsapp_handler.handle_whatsapp_request(request, db)

@app.get("/transcription", response_model=None)  # Disable response model validation
async def get_transcription(request: Request):
    transcription = twilio_whatsapp_handler.transcription
    return templates.TemplateResponse("transcription.html", {"request": request, "transcription": transcription})

# Log environment variables to ensure they're loaded correctly (masking sensitive data)
logger.info(f"TWILIO_ACCOUNT_SID: {os.getenv('TWILIO_ACCOUNT_SID')[:5]}...")
logger.info(f"TWILIO_AUTH_TOKEN: {os.getenv('TWILIO_AUTH_TOKEN')[:5]}...")
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...")