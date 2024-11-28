# message_templates.py

MESSAGE_TEMPLATES = {
    "welcome": "🎉 Hi! I'm Ada, your voice-to-text fairy! ✨🧚‍♀️\n\nYou've got 3 free transcription spells. Ready to try?\n\n🎙️ Send a voice message and watch the magic happen! 🚀",
    "welcome_with_transcription": "👋 Welcome to Ada! I see you've already sent a voice message. Great start! I'm transcribing it now and will send you the result in a moment. You have 2 more free transcriptions to try out. Enjoy the service!",
    "processing_confirmation": "🎙️ Voice message received! I'm processing it now. Your transcription will be ready in a moment. ⏳✨",
    "unsupported_media": "Oops! 😅 I can't handle {media_type} files just yet. Please send me a voice message, and I'll be happy to transcribe it for you! 🎙️✨",
    "transcription": "🎙️✨ ```YOUR TRANSCRIPT FROM ADA:```\n\n{transcription}\n--------------\n```GOT THIS FROM SOMEONE? TRY ADA! https://bit.ly/Free_Ada\u200B```",
    "long_transcription_initial": "📝 Wow, that's quite a message! It's so long it exceeds WhatsApp's limit.\n✨ No worries though - I'll craft a concise summary for you in just a moment.\n🔗 View the full transcription here: {transcription_url}",
    "long_transcription_summary": "```[SUMMARIZED WITH ADA 🧚‍♂️]```\n\n{summary}\n\n--------------\n```GOT THIS FROM SOMEONE? TRY ADA! https://bit.ly/Free_Ada\u200B```",
    "subscription_confirmation": "🔗 Great to have you on board! Your subscription is confirmed. Enjoy seamless voice transcription and all its benefits. If you have any questions or need assistance, just let me know!\n\n```MANAGE YOUR SUBSCRIPTION: https://billing.stripe.com/p/login/test_28o001fdu2lo0iQ8ww\u200B``` ⁠",
    "subscription_cancelled": "I'm sorry to see you go! 😢 Your subscription has been cancelled. If you change your mind or have any feedback, please don't hesitate to reach out. Thank you for trying me out! 🙏 - Ada",
    "subscription_reminder": "🔗 {message}",
    "ai_response": "{response}",
}

def get_message_template(template_key: str) -> str:
    """
    Retrieves a message template by key.
    :param template_key: The key for the desired template.
    :return: The message template string.
    """
    return MESSAGE_TEMPLATES.get(template_key)