from sqlalchemy import create_engine, Column, Integer, String, DateTime, ARRAY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set")

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if ENCRYPTION_KEY is None:
    raise ValueError("ENCRYPTION_KEY environment variable is not set")

fernet = Fernet(ENCRYPTION_KEY)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    encrypted_text = Column(LargeBinary)  # Changed from text to encrypted_text
    embedding = Column(ARRAY(Float))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    hash = Column(String, unique=True, index=True, nullable=True)  # Allow null values

    @property
    def text(self):
        return fernet.decrypt(self.encrypted_text).decode()

    @text.setter
    def text(self, value):
        self.encrypted_text = fernet.encrypt(value.encode())

class WhitelistedNumber(Base):
    __tablename__ = "whitelisted_numbers"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    free_trial_remaining = Column(Integer, default=3)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def get_db():
    db = SessionLocal()
    
    try:
        yield db
    finally:
        db.close()