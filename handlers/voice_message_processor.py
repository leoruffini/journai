# handlers/voice_message_processor.py

import logging
import requests
import io
from openai import OpenAI
from handlers.llm_handler import LLMHandler
from config import LLM_MODEL, TRANSCRIPTION_MODEL

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
