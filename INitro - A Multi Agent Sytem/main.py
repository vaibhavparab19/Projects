#!/usr/bin/env python3
"""
Main entry point for the INitro multi-agent system

This module demonstrates the integration of LangGraph workflow with Agent1, Agent2,
and the LangChain-powered thinking layer for campaign optimization.
"""

from langgraph_workflow import MultiAgentWorkflow, run_multi_agent_workflow
from agents.agent1 import Agent1
from agents.agent2 import Agent2
from thinking_node.core import ThinkingLayer
from datetime import datetime
import json
from sample_data import get_sample_campaign_data, get_sample_customer_data


def main():
    """Main function to demonstrate the LangGraph multi-agent system"""
    print("🚀 INitro LangGraph Multi-Agent System Starting...")
    print("=" * 60)

    # Sample campaign data using sample_data function
    campaign_data_list = get_sample_campaign_data()
    campaign_data = campaign_data_list[0].__dict__ if campaign_data_list else {
    }

    # Sample customer data using sample_data function
    customer_data = get_sample_customer_data()

    print("\n🔄 Running LangGraph Multi-Agent Workflow...")
    print("This will orchestrate Agent1, Agent2, and LangChain-powered Thinking Layer")

    # Run the LangGraph workflow
    try:
        workflow_result = run_multi_agent_workflow(
            campaign_data, customer_data)

        # Display comprehensive results
        print("\n📋 LANGGRAPH WORKFLOW RESULTS")
        print("=" * 60)

        if "error" in workflow_result:
            print(f"❌ Workflow failed: {workflow_result['error']}")
            print(f"Status: {workflow_result.get('status', 'unknown')}")
        else:
            print(
                f"✅ Workflow Status: {workflow_result.get('workflow_summary', {}).get('completion_time', 'completed')}"
            )
            print(
                f"🎯 Combined Confidence: {workflow_result.get('combined_confidence', 0.0):.2f}"
            )
            print(
                f"🤖 Agents Involved: {', '.join(workflow_result.get('workflow_summary', {}).get('agents_involved', []))}"
            )
            print(
                f"👤 Human Approval: {'Required' if workflow_result.get('workflow_summary', {}).get('human_approval_required') else 'Not Required'}"
            )

            # Display strategy summary
            unified_strategy = workflow_result.get("unified_strategy", {})
            print(f"\n📊 Unified Strategy:")
            print(f"   Approach: {unified_strategy.get('approach', 'N/A')}")
            print(
                f"   Channels: {', '.join(unified_strategy.get('channels', []))}")
            print(f"   Timeline: {unified_strategy.get('timeline', 'N/A')}")
            print(
                f"   Personalization: {unified_strategy.get('personalization', 'N/A')}"
            )

            # Display execution plan
            execution_plan = workflow_result.get("execution_plan", {})
            if execution_plan:
                print(f"\n🚀 Execution Plan:")
                timeline = execution_plan.get("execution_timeline", {})
                for week, activity in timeline.items():
                    print(f"   {week}: {activity}")

                success_metrics = execution_plan.get("success_metrics", [])
                if success_metrics:
                    print(f"\n📈 Success Metrics:")
                    for metric in success_metrics:
                        print(f"   • {metric}")

        # Save comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "workflow_type": "LangGraph Multi-Agent",
            "campaign_data": campaign_data,
            "customer_data": customer_data,
            "workflow_result": workflow_result,
            "system_info": {
                "langgraph_enabled": True,
                "langchain_integration": True,
                "openai_powered": True,
                "multi_agent_coordination": True,
            },
        }

        with open("langgraph_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n💾 Comprehensive results saved to langgraph_results.json")

    except Exception as e:
        print(f"\n❌ LangGraph Workflow Error: {str(e)}")
        print("\n🔄 Falling back to individual agent demonstration...")

        # Fallback to individual agent processing
        demonstrate_individual_agents(campaign_data, customer_data)

    print("\n🎉 INitro LangGraph Multi-Agent System completed!")
    print("=" * 60)


def demonstrate_individual_agents(campaign_data, customer_data):
    """Fallback function to demonstrate individual agents"""
    print("\n🤖 Individual Agent Demonstration")
    print("-" * 40)

    # Initialize agents
    agent1 = Agent1()
    agent2 = Agent2()
    thinking_layer = ThinkingLayer()

    # Agent 1 Processing
    print("\n📊 Agent 1: Processing campaign data...")
    try:
        agent1_result = agent1.analyze_data(
            customer_data)  # Changed to existing method
        print(
            f"✅ Agent 1 completed with {len(agent1_result['segments'])} segments")
    except Exception as e:
        print(f"❌ Agent 1 error: {str(e)}")
        agent1_result = {"error": str(e)}

    # Agent 2 Processing
    print("\n🧠 Agent 2: Processing with LangChain integration...")
    try:
        agent2_result = agent2.process_campaign_request(
            campaign_data, customer_data[0])
        print(
            f"✅ Agent 2 completed with AI confidence: {agent2_result.get('ai_confidence', 0.8):.2f}"
        )
    except Exception as e:
        print(f"❌ Agent 2 error: {str(e)}")
        agent2_result = {"error": str(e)}

    # Thinking Layer Processing
    print("\n🤔 Thinking Layer: LangChain-powered analysis...")
    try:
        thinking_result = thinking_layer.process(
            campaign_data,
            customer_data[0],
            {"agent1_output": agent1_result, "agent2_output": agent2_result},
        )
        print(
            f"✅ Thinking Layer completed with AI confidence: {thinking_result.confidence_score:.2f}"
        )
    except Exception as e:
        print(f"❌ Thinking Layer error: {str(e)}")
        thinking_result = None

    # Save individual results
    individual_results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "individual_agents_fallback",
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
        "thinking_result": (
            {
                "analysis": thinking_result.analysis if thinking_result else "Error",
                "reasoning": thinking_result.reasoning if thinking_result else "Error",
                "confidence_score": (
                    thinking_result.confidence_score if thinking_result else 0.0
                ),
            }
            if thinking_result
            else {"error": "Thinking layer failed"}
        ),
    }

    with open("individual_results.json", "w") as f:
        json.dump(individual_results, f, indent=2)

    print("\n💾 Individual results saved to individual_results.json")


if __name__ == "__main__":
    main()
