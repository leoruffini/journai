# handlers/llm_handler.py

import logging
from typing import List
from openai import OpenAI
from config import LLM_MODEL, EMBEDDING_MODEL

class LLMHandler:
    """
    Handles interactions with the Language Learning Model (LLM) API.
    """

    def __init__(self, api_key: str, model: str = LLM_MODEL):
        """
        Initializes the LLMHandler.

        :param api_key: The API key for OpenAI.
        :param model: The model name to use.
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger(f"{__name__}.LLMHandler")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding vector for the given text.

        :param text: The text to generate an embedding for.
        :return: A list of floats representing the embedding vector.
        """
        try:
            response = self.client.embeddings.create(
                input=text, 
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def generate_response(self, message: str, context: str) -> str:
        """
        Generates a response using the LLM.

        :param message: The user's message.
        :param context: Contextual information to guide the response.
        :return: The AI-generated response as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are Ada, a friendly AI assistant for voice transcription. 
Engage in brief, friendly conversation while gently steering users towards using the service or subscribing.
Keep responses under 50 words. Be responsive to the user's message, but always relate back to the transcription service.
Do not offer free trials or transcription services to users who have used all their free trials and are not subscribed.
For users with free trials, encourage them to use the service.
For subscribed users, remind them of the benefits and encourage use.
For users without free trials or subscription, focus on the benefits of subscribing."""
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nUser message: {message}\nRespond as Ada in under 50 words:"
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating AI response: {str(e)}")
            return "I'm here to help with voice transcription. How can I assist you today?"

    async def generate_summary(self, transcription: str) -> str:
        """
        Generates a summary of a transcription.

        :param transcription: The transcription text to summarize.
        :return: The summary as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are Ada, a top-tier AI assistant specializing in summarizing voice message transcriptions. Your summaries must **absolutely not exceed 1500 characters**. Before finalizing the summary, **internally count the characters** to ensure compliance. It's essential to **preserve the original tone, conversational style, and personal touches of the speaker**, including any colloquial expressions, rhetorical questions, and informal language. Maintain the original language variant and regional differences (e.g., Spain Spanish vs. Latin American Spanish) when summarizing. Do not mention the character count in your final summary."""
                    },
                    {
                        "role": "user",
                        "content": f"""Please summarize the following transcription. Ensure the summary is concise and complete, capturing the key points succinctly. It is crucial that the summary **does not exceed 1500 characters**. **Internally verify the character count** before providing the final summary. Focus on key points while **maintaining the speaker's tone, conversational style, and personal expressions**. Preserve any colloquial phrases, rhetorical questions, and regional language variants, including any regional Spanish differences:

{transcription}"""
                    }
                ],
                max_tokens=300  # Adjust as needed to ensure the summary stays under 1500 characters
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "I apologize, but I encountered an error while summarizing your message. Please try again later."