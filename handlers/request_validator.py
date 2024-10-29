# handlers/request_validator.py

import logging
from twilio.request_validator import RequestValidator
from fastapi import HTTPException

class TwilioRequestValidator:
    """
    Validates incoming Twilio requests to ensure they are legitimate.
    """

    def __init__(self, auth_token: str):
        """
        Initializes the TwilioRequestValidator.

        :param auth_token: Twilio Auth Token for request validation.
        """
        self.validator = RequestValidator(auth_token)
        self.logger = logging.getLogger(f"{__name__}.TwilioRequestValidator")

    def validate_request(self, url: str, form_data: dict, signature: str) -> bool:
        """
        Validates the incoming request using Twilio's signature.

        :param url: The URL of the incoming request.
        :param form_data: The form data from the request.
        :param signature: The Twilio signature header.
        :return: True if the request is valid, False otherwise.
        """
        is_valid = self.validator.validate(url, form_data, signature)
        if not is_valid:
            self.logger.warning("Invalid Twilio request signature.")
        return is_valid