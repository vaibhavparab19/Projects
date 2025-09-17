import random
from datetime import datetime
from typing import List, Dict, Any
from models import (
    CustomerData,
    CampaignData,
    create_sample_customer,
    create_sample_campaign,
    generate_customers,
    generate_campaigns,
)


def get_sample_campaign_data(num: int = 5) -> List[CampaignData]:
    """Generate sample campaign data"""
    return generate_campaigns(num)


def get_sample_customer_data(num: int = 50) -> List[CustomerData]:
    """Generate sample customer data"""
    return generate_customers(num)


def get_sample_context_data() -> Dict[str, Any]:
    """Generate sample historical context data"""
    return {
        "market_conditions": "normal",
        "competition_level": "medium",
        "seasonality": 1.0,
        "business_priority": "balanced",
        "historical_performance": {"previous_roi": 1.8, "previous_conversion": 0.04},
        "past_campaigns": [
            {"id": "PAST_001", "name": "Summer Sale",
                "outcome": "success", "roi": 2.1},
            {
                "id": "PAST_002",
                "name": "Holiday Promo",
                "outcome": "moderate",
                "roi": 1.5,
            },
            {
                "id": "PAST_003",
                "name": "New Year Launch",
                "outcome": "high",
                "roi": 2.8,
            },
        ],
        "behavior_patterns": {
            "high_engagement": {
                "channel": "social_media",
                "time": "evening",
                "frequency": "daily",
            },
            "medium_engagement": {
                "channel": "email",
                "time": "morning",
                "frequency": "weekly",
            },
            "low_engagement": {
                "channel": "display_ads",
                "time": "afternoon",
                "frequency": "monthly",
            },
        },
    }
