#!/usr/bin/env python3
"""
Script to run Agent 2 on sample data
"""

import json
from sample_data import get_sample_campaign_data, get_sample_customer_data
from agents.agent2 import Agent2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("🤖 Running Agent 2 on Sample Data")
    print("=" * 50)

    # Initialize Agent 2
    agent2 = Agent2()

    # Get sample data
    campaigns = get_sample_campaign_data()
    customers = get_sample_customer_data()

    # Use first campaign for demo
    campaign_data = campaigns[0]
    customer_data = customers[0]

    print(f"📊 Campaign Goals: {', '.join(campaign_data.goals)}")
    print(f"🎯 Budget: ${campaign_data.budget:,}")
    print(f"📢 Messaging: {campaign_data.messaging}")
    print(f"👥 Customer Segments: {', '.join(customer_data.segments)}")
    print("\n🔄 Processing with Agent 2...\n")

    try:
        # Process with Agent 2
        result = agent2.process_campaign_request(campaign_data, customer_data)

        print("✅ Agent 2 Processing Complete!")
        print("=" * 50)

        # Display results
        if isinstance(result, dict):
            print(f"📈 Confidence Score: {result.get('confidence', 'N/A')}")
            print(f"🎯 Strategy: {result.get('strategy', 'N/A')}")

            if "context_analysis" in result:
                context = result["context_analysis"]
                print(f"\n🧠 Context Analysis:")
                print(
                    f"   Past Campaigns: {context.get('past_campaigns_count', 0)}")
                print(
                    f"   Behavioral Insights: {len(context.get('behavioral_insights', []))}"
                )

            if "recommendations" in result:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(result["recommendations"][:3], 1):
                    print(f"   {i}. {rec}")

        # Save results
        with open("agent2_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n💾 Results saved to agent2_results.json")

    except Exception as e:
        print(f"❌ Error running Agent 2: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
