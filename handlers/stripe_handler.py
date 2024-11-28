import logging
from datetime import datetime, timezone
import stripe
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from database import WhitelistedNumber
from config import (
    STRIPE_API_KEY,
    STRIPE_WEBHOOK_SECRET,
    STRIPE_PAYMENT_LINK,
    STRIPE_CUSTOMER_PORTAL_URL,
)

logger = logging.getLogger(__name__)

class StripeHandler:
    """
    Handles Stripe payment and subscription related operations.
    """
    def __init__(self, twilio_handler=None):
        """
        Initialize the StripeHandler.

        :param twilio_handler: Optional TwilioWhatsAppHandler instance for sending notifications
        """
        self.api_key = STRIPE_API_KEY
        self.webhook_secret = STRIPE_WEBHOOK_SECRET
        self.payment_link = STRIPE_PAYMENT_LINK
        self.customer_portal_url = STRIPE_CUSTOMER_PORTAL_URL
        self.twilio_handler = twilio_handler
        stripe.api_key = self.api_key

        if not all([self.api_key, self.webhook_secret, self.payment_link, self.customer_portal_url]):
            raise ValueError("Missing required environment variables for StripeHandler")

    def create_checkout_session(self):
        """Create a new checkout session."""
        return RedirectResponse(url=self.payment_link, status_code=303)

    def construct_event(self, payload, sig_header):
        """
        Construct a Stripe event from webhook payload.

        :param payload: The webhook payload
        :param sig_header: The Stripe signature header
        :return: The constructed Stripe event
        """
        return stripe.Webhook.construct_event(payload, sig_header, self.webhook_secret)

    async def handle_checkout_completed(self, session, db: Session):
        """
        Handle a completed checkout session.

        :param session: The checkout session data
        :param db: Database session
        """
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
                    if self.twilio_handler:
                        try:
                            await self.twilio_handler.send_templated_message(phone_number, "subscription_confirmation")
                            logger.info(f"Subscription confirmation message sent to {phone_number}")
                        except Exception as e:
                            logger.error(f"Failed to send subscription confirmation message to {phone_number}: {str(e)}")
                else:
                    logger.error(f"No phone number found for customer {customer_id}")
            else:
                logger.error("No customer ID found in the checkout session")
        else:
            logger.info("Checkout completed for non-subscription product")

    async def handle_subscription_deleted(self, subscription, db: Session):
        """
        Handle a deleted subscription.

        :param subscription: The subscription data
        :param db: Database session
        """
        logger.info("Processing customer.subscription.deleted event")
        customer_id = subscription.get('customer')
        if customer_id:
            customer = stripe.Customer.retrieve(customer_id)
            phone_number = customer.get('phone')
            if phone_number:
                db.query(WhitelistedNumber).filter_by(phone_number=phone_number).delete()
                db.commit()
                logger.info(f"Removed {phone_number} from whitelist due to subscription deletion")
                if self.twilio_handler:
                    logger.info(f"Attempting to send subscription cancelled message to {phone_number}")
                    try:
                        await self.twilio_handler.send_templated_message(phone_number, "subscription_cancelled")
                        logger.info(f"Subscription cancelled message sent to {phone_number}")
                    except Exception as e:
                        logger.error(f"Failed to send subscription cancelled message to {phone_number}: {str(e)}")
                else:
                    logger.error("Twilio handler is not initialized")
            else:
                logger.error(f"No phone number found for customer {customer_id}")
        else:
            logger.error("No customer ID found in the subscription object")

    async def handle_subscription_updated(self, subscription, db: Session):
        """
        Handle an updated subscription.

        :param subscription: The subscription data
        :param db: Database session
        """
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