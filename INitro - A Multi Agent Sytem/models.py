#!/usr/bin/env python3
"""
Data models for the INitro multi-agent system

This module provides the core data structures and sample data generation functions
used throughout the system.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


class CampaignData:
    """Campaign data structure"""

    def __init__(self):
        self.goals = []
        self.budget = 0
        self.channels = []
        self.messaging = ""
        self.timing = ""
        self.name = ""
        self.description = ""
        self.target_segments = []
        self.content = {}
        self.budget_estimate = ""
        self.timeline = ""
        self.success_metrics = []
        self.optimization_score = 0
        self.evaluation_score = 0
        self.evaluation_reason = ""


class CustomerData:
    """Customer data structure"""

    def __init__(self):
        self.segments = []
        self.behavior = {}
        self.interactions = []
        self.customer_id = ""
        self.age_group = ""
        self.location = ""
        self.income_level = ""
        self.preferred_channels = []
        self.engagement_score = 0.0
        self.lifetime_value = 0.0
        self.churn_risk = 0.0


def create_sample_campaign(campaign_id: str, name: str) -> CampaignData:
    """Create a sample campaign with realistic data"""
    campaign = CampaignData()
    campaign.name = name
    campaign.description = f"Sample campaign: {name}"

    # Sample goals
    possible_goals = [
        "increase_awareness",
        "drive_sales",
        "customer_retention",
        "lead_generation",
        "brand_engagement",
    ]
    campaign.goals = random.sample(possible_goals, random.randint(1, 3))

    # Sample budget
    campaign.budget = random.randint(5000, 100000)
    campaign.budget_estimate = f"${campaign.budget:,}"

    # Sample channels
    possible_channels = [
        "Email",
        "Social Media",
        "Display Ads",
        "Search Ads",
        "Push",
        "SMS",
    ]
    campaign.channels = random.sample(possible_channels, random.randint(2, 4))

    # Sample messaging
    messages = [
        "Discover our latest innovations",
        "Limited time offer - don't miss out!",
        "Transform your experience with us",
        "Join thousands of satisfied customers",
        "Unlock exclusive benefits today",
    ]
    campaign.messaging = random.choice(messages)

    # Sample timing
    quarters = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    campaign.timing = random.choice(quarters)
    campaign.timeline = campaign.timing

    # Sample target segments
    segments = [
        "Young Professionals",
        "Tech Enthusiasts",
        "Budget Conscious",
        "Premium Customers",
        "New Subscribers",
    ]
    campaign.target_segments = random.sample(segments, random.randint(1, 3))

    # Sample content
    campaign.content = {
        "email": f"Subject: {campaign.messaging} - Check out our latest offers!",
        "social": f"🚀 {campaign.messaging} #innovation #technology",
        "display": f"Banner: {campaign.messaging}",
    }

    # Sample metrics
    metrics = [
        "Click-through rate",
        "Conversion rate",
        "Engagement score",
        "ROI",
        "Customer acquisition cost",
    ]
    campaign.success_metrics = random.sample(metrics, random.randint(2, 4))

    # Sample scores
    campaign.optimization_score = random.randint(60, 95)
    campaign.evaluation_score = random.randint(70, 100)
    campaign.evaluation_reason = f"Campaign shows strong potential with {campaign.optimization_score}% optimization score and well-defined target segments."

    return campaign


def create_sample_customer(customer_id: str, segment: str = None) -> CustomerData:
    """Create a sample customer with realistic data"""
    customer = CustomerData()
    customer.customer_id = customer_id

    # Sample segments
    possible_segments = [
        "ai_tech_enthusiast",
        "casual_tech_user",
        "professional",
        "student",
        "senior",
        "budget_conscious",
        "premium_user",
    ]
    if segment:
        customer.segments = [segment]
    else:
        customer.segments = random.sample(
            possible_segments, random.randint(1, 2))

    # Sample demographics
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    customer.age_group = random.choice(age_groups)

    locations = ["urban", "suburban", "rural"]
    customer.location = random.choice(locations)

    income_levels = ["low", "medium", "high"]
    customer.income_level = random.choice(income_levels)

    # Sample behavior
    preferred_channels = ["email", "social_media",
                          "display_ads", "push_notifications"]
    customer.preferred_channels = random.sample(
        preferred_channels, random.randint(1, 3)
    )

    customer.behavior = {
        "preferred_channel": random.choice(customer.preferred_channels),
        "engagement_time": random.choice(["morning", "afternoon", "evening", "night"]),
        "avg_order_value": random.randint(50, 500),
        "purchase_frequency": random.choice(
            ["weekly", "monthly", "quarterly", "rarely"]
        ),
        "device_preference": random.choice(["mobile", "desktop", "tablet"]),
        "content_preference": random.choice(["video", "text", "images", "interactive"]),
    }

    # Sample interactions
    possible_interactions = [
        "website_visit",
        "email_open",
        "email_click",
        "social_follow",
        "purchase",
        "support_contact",
        "app_download",
    ]
    customer.interactions = random.sample(
        possible_interactions, random.randint(2, 5))

    # Sample scores
    customer.engagement_score = round(random.uniform(0.1, 1.0), 2)
    customer.lifetime_value = round(random.uniform(100, 5000), 2)
    customer.churn_risk = round(random.uniform(0.0, 0.8), 2)

    return customer


def generate_customers(count: int = 50) -> List[CustomerData]:
    """Generate a list of sample customers"""
    customers = []
    for i in range(1, count + 1):
        customer_id = f"CUST_{i:03d}"
        customer = create_sample_customer(customer_id)
        customers.append(customer)
    return customers


def generate_campaigns(count: int = 5) -> List[CampaignData]:
    """Generate a list of sample campaigns"""
    campaigns = []
    campaign_names = [
        "Spring Launch",
        "Summer Sale",
        "Back to School",
        "Holiday Special",
        "New Year Kickoff",
        "Product Spotlight",
        "Customer Appreciation",
        "Flash Sale",
        "Brand Awareness",
        "Loyalty Rewards",
    ]

    for i in range(1, count + 1):
        campaign_id = f"CAMP_{i:03d}"
        name = campaign_names[i -
                              1] if i <= len(campaign_names) else f"Campaign {i}"
        campaign = create_sample_campaign(campaign_id, name)
        campaigns.append(campaign)

    return campaigns
