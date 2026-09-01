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
    info_website_names: Mapped[list[InfoWebsiteName]] = relationship(
        back_populates="campsite"
    )
    reviews: Mapped[list[Review]] = relationship(back_populates="campsite")
    claims: Mapped[list[Claim]] = relationship(back_populates="campsite")


class Review(Base):
    """Guest review (Google for now). Stars, author, and full text live here."""

    __tablename__ = "reviews"
    __table_args__ = (Index("reviews_campsite_idx", "campsite_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    campsite_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_uid: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    campsite: Mapped[Campsite] = relationship(back_populates="reviews")
    claims: Mapped[list[Claim]] = relationship(back_populates="review")


class Claim(Base):
    """Atomic site fact split from a review. No stars; join `reviews` for that.

    Readable join for manual review: view claims_with_reviews.
    """

    __tablename__ = "claims"
    __table_args__ = (
        Index("claim_campsite_idx", "campsite_id"),
        Index("claim_review_idx", "review_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    review_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False
    )
    campsite_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    claim: Mapped[str | None] = mapped_column(Text)
    evidence_span: Mapped[str | None] = mapped_column(Text)
    polarity: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    # Qwen3-Embedding-8B via Nebius with dimensions=1536 (HNSW max is 2000).
    embedding = mapped_column(Vector(1536), nullable=True)

    review: Mapped[Review] = relationship(back_populates="claims")
    campsite: Mapped[Campsite] = relationship(back_populates="claims")


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
    info_website_name_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("info_website_names.id", ondelete="SET NULL"),
        nullable=True,
    )
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
    info_website_name: Mapped[InfoWebsiteName | None] = relationship(
        back_populates="accommodation_types"
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


class InfoWebsiteName(Base):
    """Parks.org.il rate-card lodging product (name only)."""

    __tablename__ = "info_website_names"
    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "name",
            name="info_website_names_site_id_name_key",
        ),
        Index("info_website_names_site_idx", "site_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)

    campsite: Mapped[Campsite] = relationship(back_populates="info_website_names")
    list_prices: Mapped[list[ListPrice]] = relationship(
        back_populates="info_website_name"
    )
    accommodation_types: Mapped[list[AccommodationType]] = relationship(
        back_populates="info_website_name"
    )


class ListPrice(Base):
    """Published parks.org.il rate-card prices (not date-specific availability)."""

    __tablename__ = "list_prices"
    __table_args__ = (
        UniqueConstraint(
            "info_website_name_id",
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
    info_website_name_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("info_website_names.id", ondelete="RESTRICT"),
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
    info_website_name: Mapped[InfoWebsiteName] = relationship(
        back_populates="list_prices"
    )
