from services.fraud.features.base import Entity
from services.fraud.features.registry import register_static

register_static(
    name="email_emails_match_score",
    entity=Entity.CUSTOMER_ID,
    feature_view="customer_features",
    group="customer",
)
