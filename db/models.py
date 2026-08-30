"""SQLAlchemy models — source of truth for schema (Alembic migrations)."""

from __future__ import annotations

from datetime import date, datetime, time

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
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    accommodation_types: Mapped[list[AccommodationType]] = relationship(
        back_populates="campsite"
    )
    notices: Mapped[list[Notice]] = relationship(back_populates="campsite")
    list_prices: Mapped[list[ListPrice]] = relationship(back_populates="campsite")


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
    # Qwen3-Embedding-8B via Nebius with dimensions=1536 (HNSW max is 2000).
    embedding = mapped_column(Vector(1536), nullable=True)


class Amenity(Base):
    __tablename__ = "amenities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Qwen3-Embedding-8B via Nebius with dimensions=1536 (HNSW max is 2000).
    embedding = mapped_column(Vector(1536), nullable=True)


class Notice(Base):
    """Ephemeral official-site notices (outages, temporary closures).

    Lifecycle: upsert while the stored HTML element is still on the page;
    delete the row when the next scrape no longer finds that element.
    """

    __tablename__ = "notices"
    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "html_element_sha256",
            name="notices_site_element_key",
        ),
        Index("notices_site_idx", "site_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    page_url: Mapped[str | None] = mapped_column(Text)
    lang: Mapped[str | None] = mapped_column(Text)
    notice_he: Mapped[str | None] = mapped_column(Text)
    notice_en: Mapped[str | None] = mapped_column(Text)
    # Exact HTML node that carried the notice; missing on next scrape ⇒ delete.
    html_element: Mapped[str] = mapped_column(Text, nullable=False)
    html_element_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    # Qwen3-Embedding-8B via Nebius with dimensions=1536 (HNSW max is 2000).
    embedding = mapped_column(Vector(1536), nullable=True)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    campsite: Mapped[Campsite] = relationship(back_populates="notices")


class AccommodationType(Base):
    __tablename__ = "accommodation_types"
    __table_args__ = (
        UniqueConstraint(
            "hotel_id",
            "name",
            name="accommodation_types_hotel_id_name_key",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hotel_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # Raw Hebrew tooltip text sent to the extraction LLM.
    description: Mapped[str | None] = mapped_column(Text)
    # JSONB array of amenities.id values, e.g. [1, 5, 12]
    amenities = mapped_column(JSONB)
    # JSONB array of amenities.id values that are explicitly not included
    not_included = mapped_column(JSONB)
    # Readable join of amenity ids → names: view accommodation_types_with_amenity_names
    max_occupancy: Mapped[int | None] = mapped_column(Integer)
    total_beds: Mapped[int | None] = mapped_column(Integer)
    # Connected rooms/units in this listing (e.g. 2 for "שתי חושות מחוברות"). Default 1.
    room_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # e.g. {"double_beds": 1, "single_beds": 2}
    bed_configuration = mapped_column(JSONB)
    # Up to 3 absolute image URLs from the booking .imageholder gallery
    image_urls = mapped_column(JSONB)
    check_in_time: Mapped[time | None] = mapped_column(Time)
    check_out_time: Mapped[time | None] = mapped_column(Time)
    # e.g. {"min_weekend_nights": 2, "min_holiday_nights": 2, "pets_allowed": false}
    policy_rules = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    campsite: Mapped[Campsite] = relationship(back_populates="accommodation_types")
    availability: Mapped[list[Availability]] = relationship(
        back_populates="accommodation_type"
    )
    list_prices: Mapped[list[ListPrice]] = relationship(
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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    campsite: Mapped[Campsite] = relationship(back_populates="availability")
    accommodation_type: Mapped[AccommodationType] = relationship(
        back_populates="availability"
    )


class ListPrice(Base):
    """Published parks.org.il rate-card prices (not date-specific availability)."""

    __tablename__ = "list_prices"
    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "accommodation_type_id",
            "guest_type",
            "rate_period",
            "rate_class",
            name="list_prices_unique_rate",
        ),
        Index("list_prices_site_idx", "site_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    accommodation_type_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("accommodation_types.id", ondelete="RESTRICT"),
        nullable=False,
    )
    guest_type: Mapped[str] = mapped_column(Text, nullable=False)
    rate_period: Mapped[str] = mapped_column(Text, nullable=False)
    rate_class: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(Text, nullable=False, default="ILS")
    notes: Mapped[str | None] = mapped_column(Text)
    raw_label: Mapped[str] = mapped_column(Text, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    campsite: Mapped[Campsite] = relationship(back_populates="list_prices")
    accommodation_type: Mapped[AccommodationType] = relationship(
        back_populates="list_prices"
    )
