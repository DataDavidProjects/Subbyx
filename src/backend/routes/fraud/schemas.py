from pydantic import BaseModel


class SegmentDetermineRequest(BaseModel):
    email: str
    fiscal_code: str | None = None
    card_fingerprint: str | None = None


class SegmentDetermineResponse(BaseModel):
    segment: str
    reason: str


class FeaturesGetRequest(BaseModel):
    customer_id: str
    email: str
    checkout_data: dict
    segment: str


class FeaturesGetResponse(BaseModel):
    features: dict


class BlacklistCheckRequest(BaseModel):
    email: str


class BlacklistCheckResponse(BaseModel):
    triggered: bool
    rule: str
    reason: str


class ScoreRequest(BaseModel):
    features: dict


class ScoreResponse(BaseModel):
    score: float
    production_score: float | None = None
    shadow_score: float | None = None
    canary_score: float | None = None
    scored_by: str | None = None


class DecisionRequest(BaseModel):
    score: float
    segment: str


class DecisionResponse(BaseModel):
    decision: str
    reason: str


class CheckoutRequest(BaseModel):
    customer_id: str
    email: str
    checkout_data: dict
    timestamp: str | None = None
    customer_name: str | None = None
    document_name: str | None = None
    account_name: str | None = None
    has_high_end_device: bool | None = None
    fiscal_code: str | None = None
    gender: str | None = None
    birth_date: str | None = None
    birth_province: str | None = None
    birth_country: str | None = None
    card_fingerprint: str | None = None


class CheckoutResponse(BaseModel):
    decision: str
    reason: str
    rule_triggered: str | None
    score: float | None
    segment: str | None
    segment_reason: str | None
    features: dict | None = None
    production_score: float | None = None
    shadow_score: float | None = None
    canary_score: float | None = None
    scored_by: str | None = None
