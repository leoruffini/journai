# message_templates.py

MESSAGE_TEMPLATES = {
    "welcome": "🎉 Hi! I'm Ada, your voice-to-text fairy! ✨🧚‍♀️\n\nYou've got 3 free transcription spells. Ready to try?\n\n🎙️ Send a voice message and watch the magic happen! 🚀",
    "welcome_with_transcription": "👋 Welcome to Ada! I see you've already sent a voice message. Great start! I'm transcribing it now and will send you the result in a moment. You have 2 more free transcriptions to try out. Enjoy the service!",
    "processing_confirmation": "🎙️ Voice message received! I'm processing it now. Your transcription will be ready in a moment. ⏳✨",
    "subscription_cancelled": "I'm sorry to see you go! 😢 Your subscription has been cancelled. If you change your mind or have any feedback, please don't hesitate to reach out. Thank you for trying me out! 🙏 - Ada",
    "unsupported_media": "Oops! 😅 I can't handle {media_type} files just yet. Please send me a voice message, and I'll be happy to transcribe it for you! 🎙️✨",
    "text_message_with_free_trials": "Hi there! 👋 I can only transcribe voice messages. 🎙️✨ You have {free_trial_remaining} free transcriptions left. Please send a voice message to use them.",
    "subscribed_user_text_message": "Hi there! 👋 I can only transcribe voice messages. 🎙️✨ Please send a voice message to get started.",
    "transcription": "🎙️✨ ```YOUR TRANSCRIPT FROM ADA:```\n\n{transcription}\n--------------\n```GOT THIS FROM SOMEONE? TRY ADA! https://bit.ly/Free_Ada\u200B```",
    "long_transcription_initial": "📝 Wow, that's quite a message! It's so long it exceeds WhatsApp's limit.\n✨ No worries though - I'll craft a concise summary for you in just a moment.\n🔗 View the full transcription here: {transcription_url}",
    "long_transcription_summary": "```[SUMMARIZED WITH ADA 🧚‍♂️]```\n\n{summary}\n\n--------------\n```GOT THIS FROM SOMEONE? TRY ADA! https://bit.ly/Free_Ada\u200B```",
    "subscription_reminder": "🔗 {message}",
    "ai_response": "{response}",
    "subscription_confirmation": "🔗 {response}",
}

def get_message_template(template_key: str) -> str:
    """
    Retrieves a message template by key.
    :param template_key: The key for the desired template.
    :return: The message template string.
    """
    return MESSAGE_TEMPLATES.get(template_key)