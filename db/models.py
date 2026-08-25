"""SQLAlchemy models — source of truth for schema (Alembic migrations)."""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Campsite(Base):
    __tablename__ = "campsites"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    booking_hotel_id: Mapped[str | None] = mapped_column(Text, unique=True)

    availability: Mapped[list[Availability]] = relationship(back_populates="campsite")


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (Index("claim_campsite_idx", "campsite_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    campsite_id: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    review_author: Mapped[str | None] = mapped_column(Text)
    review_date: Mapped[str | None] = mapped_column(Text)
    lang: Mapped[str | None] = mapped_column(Text)
    claim_he: Mapped[str | None] = mapped_column(Text)
    claim_en: Mapped[str | None] = mapped_column(Text)
    evidence_span: Mapped[str | None] = mapped_column(Text)
    polarity: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    claim_uid: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    # HNSW index is created in the Alembic migration (pgvector).
    embedding = mapped_column(Vector(1536), nullable=True)


class AccommodationType(Base):
    __tablename__ = "accommodation_types"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    availability: Mapped[list[Availability]] = relationship(
        back_populates="accommodation_type"
    )


class Availability(Base):
    __tablename__ = "availability"
    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "start_date",
            "end_date",
            "accommodation_type_id",
            "adults_no",
            name="availability_unique_slot",
        ),
        Index("availability_site_dates_idx", "site_id", "start_date", "end_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    accommodation_type_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("accommodation_types.id", ondelete="RESTRICT"),
        nullable=False,
    )
    price: Mapped[float] = mapped_column(Float, nullable=False)
    adults_no: Mapped[int] = mapped_column(Integer, nullable=False)
    room_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    campsite: Mapped[Campsite] = relationship(back_populates="availability")
    accommodation_type: Mapped[AccommodationType] = relationship(
        back_populates="availability"
    )
