from pydantic import BaseModel


class FeaturesGetResponse(BaseModel):
    features: dict
