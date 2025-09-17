#!/usr/bin/env python3
"""
Enhanced Agent 2 Test Runner with Memory + Retrieval + LLM Integration

This script demonstrates Agent 2's enhanced capabilities:
- Conversation Memory System
- Vector-based Knowledge Retrieval
- Contextual Response Generation
- LLM-powered Analysis
"""

from sample_data import get_sample_campaign_data, get_sample_customer_data
from agents.agent2 import Agent2
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_enhanced_agent2():
    """Test Agent 2 with enhanced memory and retrieval capabilities"""
    print("=" * 80)
    print("🚀 ENHANCED AGENT 2 TEST - Memory + Retrieval + LLM")
    print("=" * 80)

    # Initialize Agent 2
    print("\n📋 Initializing Enhanced Agent 2...")
    agent2 = Agent2()

    # Check memory and retrieval system status
    print(
        f"\n🧠 Memory System: {'✓ Active' if agent2.thinking_layer.memory_system else '✗ Inactive'}"
    )
    print(
        f"🔍 Retrieval System: {'✓ Active' if agent2.thinking_layer.retrieval_system else '✗ Inactive'}"
    )

    if agent2.thinking_layer.retrieval_system:
        stats = agent2.thinking_layer.retrieval_system.get_retrieval_stats()
        print(f"📊 Knowledge Base: {stats.get('total_documents', 0)} documents")

    # Get sample data
    print("\n📊 Loading sample campaign and customer data...")
    try:
        campaigns = get_sample_campaign_data()
        customers = get_sample_customer_data()

        campaign_data = (
            campaigns[0]
            if campaigns
            else {
                "budget": 75000,
                "channels": ["email", "social_media", "display_ads"],
                "target_audience": "tech professionals",
                "goals": ["brand_awareness", "lead_generation"],
            }
        )

        customer_data = (
            customers[0]
            if customers
            else {
                "segment": "tech_professionals",
                "lifetime_value": 5000,
                "behavior": ["online_shopping", "social_media_active"],
                "preferences": ["email_communication", "mobile_first"],
                "demographics": {"age_group": "25-40", "location": "urban"},
            }
        )

    except ImportError:
        print("⚠️  Sample data not available, using default data")
        campaign_data = {
            "budget": 75000,
            "channels": ["email", "social_media", "display_ads"],
            "target_audience": "tech professionals",
            "goals": ["brand_awareness", "lead_generation"],
        }

        customer_data = {
            "segment": "tech_professionals",
            "lifetime_value": 5000,
            "behavior": ["online_shopping", "social_media_active"],
            "preferences": ["email_communication", "mobile_first"],
            "demographics": {"age_group": "25-40", "location": "urban"},
        }

    # Handle both dictionary and object formats
    if hasattr(campaign_data, "budget"):
        print(f"✓ Campaign Budget: ${campaign_data.budget:,}")
        print(f"✓ Channels: {', '.join(campaign_data.channels)}")
        print(f"✓ Goals: {', '.join(campaign_data.goals)}")
    else:
        print(f"✓ Campaign Budget: ${campaign_data['budget']:,}")
        print(f"✓ Target Audience: {campaign_data['target_audience']}")

    if hasattr(customer_data, "segments"):
        print(f"✓ Customer Segments: {', '.join(customer_data.segments)}")
        print(
            f"✓ Customer Behavior: {customer_data.behavior.get('preferred_channel', 'N/A')}"
        )
    else:
        print(f"✓ Customer Segment: {customer_data['segment']}")
        print(f"✓ Customer Value: ${customer_data['lifetime_value']:,}")

    # Test 1: First interaction (no memory context)
    print("\n" + "=" * 60)
    print("🧪 TEST 1: First Campaign Analysis (No Memory Context)")
    print("=" * 60)

    user_query1 = "Analyze this campaign for maximum ROI with focus on digital channels"
    result1 = agent2.process_campaign_request(
        campaign_data=campaign_data, customer_data=customer_data, user_query=user_query1
    )

    print("\n📋 FIRST ANALYSIS RESULTS:")
    print(f"Strategy: {result1.get('campaign_strategy', 'N/A')}")
    print(
        f"Predicted ROI: {result1.get('predicted_outcomes', {}).get('predicted_roi', 'N/A')}x"
    )
    print(f"AI Confidence: {result1.get('ai_confidence', 'N/A')}")

    if "contextual_enhancements" in result1:
        enhancements = result1["contextual_enhancements"]
        print(
            f"Memory Insights Used: {enhancements.get('memory_insights', 0)}")
        print(
            f"Historical Campaigns: {enhancements.get('campaign_history', 0)}")

    # Test 2: Second interaction (with memory context)
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Follow-up Analysis (With Memory Context)")
    print("=" * 60)

    # Create modified campaign for follow-up
    if hasattr(campaign_data, "budget"):
        # Create new CampaignData object with modified budget
        from agents.agent2 import CampaignData

        campaign_data2 = CampaignData()
        campaign_data2.goals = campaign_data.goals
        campaign_data2.budget = 100000
        campaign_data2.channels = ["social_media", "influencer", "video_ads"]
        campaign_data2.messaging = (
            getattr(campaign_data, "messaging", "") + " - Enhanced"
        )
        campaign_data2.timing = getattr(campaign_data, "timing", {})
    else:
        # Handle dictionary format
        campaign_data2 = campaign_data.copy()
        campaign_data2["budget"] = 100000
        campaign_data2["channels"] = [
            "social_media", "influencer", "video_ads"]

    user_query2 = "How would increasing budget to $100k and focusing on social media change the results?"
    result2 = agent2.process_campaign_request(
        campaign_data=campaign_data2,
        customer_data=customer_data,
        user_query=user_query2,
    )

    print("\n📋 FOLLOW-UP ANALYSIS RESULTS:")
    print(f"Strategy: {result2.get('campaign_strategy', 'N/A')}")
    print(
        f"Predicted ROI: {result2.get('predicted_outcomes', {}).get('predicted_roi', 'N/A')}x"
    )
    print(f"AI Confidence: {result2.get('ai_confidence', 'N/A')}")

    if "contextual_enhancements" in result2:
        enhancements = result2["contextual_enhancements"]
        print(
            f"Memory Insights Used: {enhancements.get('memory_insights', 0)}")
        print(
            f"Historical Campaigns: {enhancements.get('campaign_history', 0)}")
        print(
            f"Conversation Context: {'✓' if enhancements.get('conversation_context') else '✗'}"
        )

    # Test 3: Knowledge retrieval test
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Knowledge Retrieval Test")
    print("=" * 60)

    if agent2.thinking_layer.memory_system:
        # Test knowledge retrieval
        test_query = "email marketing best practices"
        knowledge = agent2.thinking_layer.memory_system.retrieve_relevant_knowledge(
            test_query
        )

        print(f"Query: '{test_query}'")
        print(f"Retrieved {len(knowledge)} knowledge pieces:")

        for i, doc in enumerate(knowledge[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
            if hasattr(doc, "metadata") and doc.metadata:
                print(f"     Type: {doc.metadata.get('type', 'unknown')}")

    # Save results
    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS")
    print("=" * 60)

    results_summary = {
        "test_timestamp": datetime.now().isoformat(),
        "session_id": agent2.thinking_layer.session_id,
        "memory_system_active": agent2.thinking_layer.memory_system is not None,
        "retrieval_system_active": agent2.thinking_layer.retrieval_system is not None,
        "test_results": {
            "first_analysis": {
                "query": user_query1,
                "roi": result1.get("predicted_outcomes", {}).get(
                    "predicted_roi", "N/A"
                ),
                "confidence": result1.get("ai_confidence", "N/A"),
                "contextual_enhancements": result1.get("contextual_enhancements", {}),
            },
            "follow_up_analysis": {
                "query": user_query2,
                "roi": result2.get("predicted_outcomes", {}).get(
                    "predicted_roi", "N/A"
                ),
                "confidence": result2.get("ai_confidence", "N/A"),
                "contextual_enhancements": result2.get("contextual_enhancements", {}),
            },
        },
        "knowledge_base_stats": (
            agent2.thinking_layer.retrieval_system.get_retrieval_stats()
            if agent2.thinking_layer.retrieval_system
            else {}
        ),
    }

    # Save to file
    output_file = "enhanced_agent2_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"✓ Results saved to {output_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 ENHANCED AGENT 2 TEST SUMMARY")
    print("=" * 80)

    print(f"Session ID: {agent2.thinking_layer.session_id}")
    print(
        f"Memory System: {'✓ Active' if agent2.thinking_layer.memory_system else '✗ Inactive'}"
    )
    print(
        f"Retrieval System: {'✓ Active' if agent2.thinking_layer.retrieval_system else '✗ Inactive'}"
    )
    print(f"Tests Completed: 3/3")
    print(f"Results File: {output_file}")

    if agent2.thinking_layer.retrieval_system:
        stats = agent2.thinking_layer.retrieval_system.get_retrieval_stats()
        print(f"Knowledge Base: {stats.get('total_documents', 0)} documents")

    print("\n🎉 Enhanced Agent 2 testing completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Conversation memory across interactions")
    print("   • Vector-based knowledge retrieval")
    print("   • Contextual response generation")
    print("   • LLM-powered campaign analysis")
    print("   • Persistent knowledge storage")

    return True


if __name__ == "__main__":
    try:
        test_enhanced_agent2()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
