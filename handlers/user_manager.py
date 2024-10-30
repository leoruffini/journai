from sqlalchemy.orm import Session
from database import User, WhitelistedNumber
from datetime import datetime, timezone

class UserManager:
    def __init__(self, db: Session):
        self.db = db

    def get_user_by_phone(self, phone_number: str) -> User:
        return self.db.query(User).filter_by(phone_number=phone_number).first()

    def create_user(self, phone_number: str) -> User:
        user = User(phone_number=phone_number)
        self.db.add(user)
        self.db.commit()
        return user

    def is_whitelisted(self, phone_number: str) -> bool:
        whitelisted = self.db.query(WhitelistedNumber).filter(
            WhitelistedNumber.phone_number == phone_number,
            WhitelistedNumber.expires_at > datetime.now(timezone.utc)
        ).first()
        return whitelisted is not None

    def update_free_trial(self, user: User):
        if user.free_trial_remaining > 0:
            user.free_trial_remaining -= 1
            self.db.commit()