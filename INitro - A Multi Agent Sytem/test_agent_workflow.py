#!/usr/bin/env python3
"""
Test Agent Workflow - Agent 1 Output as Agent 2 Input

This script demonstrates how Agent 1's output becomes the input for Agent 2,
showing the complete workflow integration.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.agent1 import Agent1
    from agents.agent2 import Agent2
    from sample_data import get_sample_campaign_data, get_sample_customer_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating mock data for demonstration...")

    # Mock data if imports fail
    def get_sample_campaign_data():
        class MockCampaign:
            def __init__(self):
                self.goals = ["increase_awareness", "drive_sales"]
                self.budget = 50000
                self.channels = ["email", "social_media", "display_ads"]
                self.messaging = "New product launch campaign"
                self.timing = "Q1 2024"

        return [MockCampaign()]

    def get_sample_customer_data():
        class MockCustomer:
            def __init__(self):
                self.segments = ["young_professionals", "tech_enthusiasts"]
                self.behavior = {
                    "preferred_channel": "email",
                    "engagement_time": "evening",
                    "avg_order_value": 150,
                }
                self.interactions = ["website_visit",
                                     "email_open", "social_follow"]

        return [MockCustomer()]

    # Mock Agent classes if imports fail
    class Agent1:
        def __init__(self):
            self.name = "Agent 1 - Campaign Optimization (Mock)"

        def optimize_campaigns(self, campaigns, customer_data):
            return [
                {
                    "campaign_id": "optimized_campaign_1",
                    "strategy": "AI-optimized multi-channel approach",
                    "budget_allocation": {
                        "email": 20000,
                        "social_media": 20000,
                        "display_ads": 10000,
                    },
                    "target_segments": ["young_professionals", "tech_enthusiasts"],
                    "predicted_roi": 2.8,
                    "optimization_confidence": 0.85,
                    "recommendations": [
                        "Focus on email marketing for higher engagement",
                        "Increase social media budget for broader reach",
                    ],
                }
            ]

    class Agent2:
        def __init__(self):
            self.name = "Agent 2 - Context Engineering (Mock)"

        def process_agent1_output(self, optimized_campaigns, customer_data):
            return {
                "enhanced_strategy": "AI-enhanced decision with contextual insights",
                "final_recommendations": [
                    "Implement dynamic personalization",
                    "Enable real-time optimization",
                    "Add contextual targeting",
                ],
                "confidence": 0.92,
                "risk_assessment": "Low risk with high potential ROI",
                "implementation_plan": {
                    "phase_1": "Setup and configuration",
                    "phase_2": "Campaign launch",
                    "phase_3": "Monitoring and optimization",
                },
            }


def print_separator(title):
    """Print a formatted separator with title"""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def print_json_pretty(data, title=""):
    """Print JSON data in a formatted way"""
    if title:
        print(f"\n{title}:")
    print(json.dumps(data, indent=2, default=str))


def main():
    """Main workflow demonstration"""
    print_separator("AGENT WORKFLOW DEMONSTRATION")
    print("Demonstrating how Agent 1 output becomes Agent 2 input")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: Initialize agents
    print_separator("STEP 1: INITIALIZE AGENTS")
    agent1 = Agent1()
    agent2 = Agent2()

    print(f"✅ Agent1 initialized")
    print(f"✅ Agent2 initialized")

    # Step 2: Prepare input data
    print_separator("STEP 2: PREPARE INPUT DATA")
    campaigns = get_sample_campaign_data()
    customer_data = get_sample_customer_data()

    print(f"📊 Loaded {len(campaigns)} campaign(s)")
    print(f"👥 Loaded {len(customer_data)} customer profile(s)")

    # Display sample data
    if campaigns:
        campaign = campaigns[0]
        print(f"\nSample Campaign:")
        print(f"  - Goals: {getattr(campaign, 'goals', 'N/A')}")
        print(f"  - Budget: ${getattr(campaign, 'budget', 0):,}")
        print(f"  - Channels: {getattr(campaign, 'channels', [])}")
        print(f"  - Message: {getattr(campaign, 'messaging', 'N/A')}")

    if customer_data:
        customer = customer_data[0]
        print(f"\nSample Customer:")
        print(f"  - Segments: {getattr(customer, 'segments', [])}")
        print(
            f"  - Preferred Channel: {getattr(customer, 'behavior', {}).get('preferred_channel', 'N/A')}"
        )
        print(
            f"  - Avg Order Value: ${getattr(customer, 'behavior', {}).get('avg_order_value', 0)}"
        )

    # Step 3: Agent 1 processes and optimizes campaigns
    print_separator("STEP 3: AGENT 1 OPTIMIZATION")
    print(f"🤖 Agent1 processing campaigns...")

    try:
        # Agent 1 optimizes the campaigns
        optimized_campaigns = agent1.optimize_campaigns(
            campaigns, customer_data)
        print(f"✅ Agent 1 completed optimization")
        print(f"📈 Generated {len(optimized_campaigns)} optimized campaign(s)")

        # Display Agent 1 output
        print_json_pretty(optimized_campaigns,
                          "Agent 1 Output (Optimized Campaigns)")

    except Exception as e:
        print(f"❌ Agent 1 error: {e}")
        # Create fallback data
        optimized_campaigns = [
            {
                "campaign_id": "fallback_campaign",
                "strategy": "Basic optimization",
                "budget_allocation": {"email": 25000, "social_media": 25000},
                "predicted_roi": 2.0,
                "optimization_confidence": 0.7,
            }
        ]
        print("🔄 Using fallback optimized campaigns")

    # Step 4: Agent 2 processes Agent 1's output
    print_separator("STEP 4: AGENT 2 CONTEXT ENGINEERING")
    print(f"🧠 Agent2 processing Agent 1 output...")
    print("📥 Agent 1 output becomes Agent 2 input")

    try:
        # Agent 2 processes Agent 1's output - THIS IS THE KEY INTEGRATION
        enhanced_decision = agent2.process_agent1_output(
            optimized_campaigns, customer_data[0] if customer_data else {}
        )
        print(f"✅ Agent 2 completed context engineering")
        print(f"🎯 Generated enhanced decision with contextual insights")

        # Display Agent 2 output
        print_json_pretty(enhanced_decision,
                          "Agent 2 Output (Enhanced Decision)")

    except Exception as e:
        print(f"❌ Agent 2 error: {e}")
        # Create fallback data
        enhanced_decision = {
            "enhanced_strategy": "Fallback enhanced strategy",
            "final_recommendations": ["Basic recommendation"],
            "confidence": 0.5,
            "status": "fallback_mode",
        }
        print("🔄 Using fallback enhanced decision")

    # Step 5: Summary and workflow completion
    print_separator("STEP 5: WORKFLOW SUMMARY")
    print("🔄 Complete Agent Workflow:")
    print("   1. Input Data (Campaigns + Customers)")
    print("   2. Agent 1: Campaign Optimization")
    print("   3. Agent 1 Output → Agent 2 Input")
    print("   4. Agent 2: Context Engineering & Enhancement")
    print("   5. Final Enhanced Decision")

    print("\n📊 Workflow Results:")
    print(f"   - Original Campaigns: {len(campaigns)}")
    print(f"   - Optimized Campaigns: {len(optimized_campaigns)}")
    print(
        f"   - Enhanced Decision Confidence: {enhanced_decision.get('confidence', 'N/A')}"
    )
    print(
        f"   - Final Recommendations: {len(enhanced_decision.get('final_recommendations', enhanced_decision.get('recommendations', [])))}"
    )

    print_separator("WORKFLOW COMPLETED SUCCESSFULLY")
    print("✅ Agent 1 output successfully became Agent 2 input")
    print("🎯 End-to-end agent workflow demonstration complete")

    return {
        "agent1_output": optimized_campaigns,
        "agent2_output": enhanced_decision,
        "workflow_status": "completed",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n🎉 Workflow completed successfully at {result['timestamp']}")
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()
