"""Agent 2 - Context Engineering & Knowledge Base with LangChain Integration

This agent manages context engineering and knowledge base operations,
using LangChain-powered Agent Thinking Layer with AI-driven Analysis, Reasoning, Preference, and Simulation nodes.
"""

from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
import uuid
import time
from dataclasses import dataclass, asdict
from thinking_node.core import ThinkingLayer, ThinkingResult
from tools.llm_service import LLMService
from tools.memory_system import MemorySystem
from tools.retrieval_system import RetrievalSystem
from config import Config
from agents.agent1 import Agent1  # Import Agent1 if needed for type hints


class CampaignData:
    """Campaign data structure"""

    def __init__(self):
        self.goals = []
        self.budget = 0
        self.channels = []
        self.messaging = ""
        self.timing = ""


class CustomerData:
    """Customer data structure"""

    def __init__(self):
        self.segments = []
        self.behavior = {}
        self.interactions = []


class ContextStore:
    """Context store for past campaigns and outcomes"""

    def __init__(self):
        self.past_campaigns = []
        self.behavior_patterns = {}
        self.outcomes = []

    def store_campaign(self, campaign, outcome):
        """Store campaign and its outcome"""
        self.past_campaigns.append(campaign)
        self.outcomes.append(outcome)

    def get_relevant_context(self, current_campaign):
        """Get relevant context for current campaign"""
        return {
            "similar_campaigns": self.past_campaigns[-5:],  # Last 5 campaigns
            "success_patterns": self.behavior_patterns,
        }


class AgentThinkingLayer:
    """LangChain-powered Agent thinking layer with AI nodes"""

    def __init__(self):
        self.config = Config()
        self.llm_service = LLMService()
        self.thinking_layer = ThinkingLayer()  # LangChain-powered thinking layer

        # Initialize memory and retrieval systems
        self.memory_system = MemorySystem(
            llm=self.llm_service.llm, persist_directory="./agent2_memory"
        )

        if self.memory_system.vector_store:
            self.retrieval_system = RetrievalSystem(
                llm=self.llm_service.llm, vector_store=self.memory_system.vector_store
            )
        else:
            self.retrieval_system = None

        self.session_id = str(uuid.uuid4())

    def process(self, campaign_data, customer_data, context):
        """Process through LangChain-powered thinking nodes"""
        # Use LangChain thinking layer for AI-powered processing
        thinking_result = self.thinking_layer.process(
            campaign_data, customer_data, context
        )

        return {
            "predicted_roi": thinking_result.simulation.get(
                "predicted_outcomes", {}
            ).get("roi_estimate", 2.5),
            "expected_reach": thinking_result.simulation.get(
                "predicted_outcomes", {}
            ).get("reach_estimate", 10000),
            "conversion_rate": thinking_result.simulation.get(
                "predicted_outcomes", {}
            ).get("conversion_rate", 0.05),
            "simulation_confidence": thinking_result.confidence_score,
            "recommendations": thinking_result.recommendations,
            "ai_insights": {
                "analysis": thinking_result.analysis,
                "reasoning": thinking_result.reasoning,
                "preferences": thinking_result.preferences,
                "simulation": thinking_result.simulation,
            },
        }


# Legacy node classes maintained for compatibility
class AnalysisNode:
    """Analysis node for data processing"""

    def analyze(self, campaign_data, customer_data):
        """Analyze campaign and customer data"""
        return {
            "target_segments": customer_data.segments,
            "budget_allocation": self._allocate_budget(campaign_data),
            "channel_effectiveness": self._analyze_channels(campaign_data.channels),
        }

    def _allocate_budget(self, campaign_data):
        """Allocate budget across channels"""
        return {
            channel: campaign_data.budget / len(campaign_data.channels)
            for channel in campaign_data.channels
        }

    def _analyze_channels(self, channels):
        """Analyze channel effectiveness"""
        return {channel: 0.8 for channel in channels}  # Mock effectiveness


class ReasoningNode:
    """Reasoning node for logical processing"""

    def reason(self, analysis_result, context):
        """Apply reasoning based on analysis and context"""
        recommendations = []

        # Use context to improve recommendations
        if context.get("similar_campaigns"):
            recommendations.append("Apply learnings from similar campaigns")

        # Budget optimization reasoning
        if analysis_result.get("budget_allocation"):
            recommendations.append(
                "Optimize budget allocation based on channel effectiveness"
            )

        return {
            "recommendations": recommendations,
            "confidence_score": 0.85,
            "risk_factors": ["Market volatility", "Seasonal trends"],
        }


class PreferenceNode:
    """Preference node for applying business rules"""

    def apply_preferences(self, reasoning_result):
        """Apply business preferences and constraints"""
        preferences = {
            "preferred_channels": ["email", "social_media"],
            "risk_tolerance": "medium",
            "compliance_requirements": ["GDPR", "CAN-SPAM"],
        }

        # Apply preferences to recommendations
        filtered_recommendations = []
        for rec in reasoning_result.get("recommendations", []):
            if self._meets_preferences(rec, preferences):
                filtered_recommendations.append(rec)

        return {
            "filtered_recommendations": filtered_recommendations,
            "preference_score": 0.9,
            "compliance_status": "compliant",
        }

    def _meets_preferences(self, recommendation, preferences):
        """Check if recommendation meets preferences"""
        return True  # Simplified logic


class SimulationNode:
    """Simulation node for outcome prediction"""

    def simulate(self, preference_result):
        """Simulate campaign outcomes"""
        return {
            "predicted_roi": 2.5,
            "expected_reach": 10000,
            "conversion_rate": 0.05,
            "simulation_confidence": 0.8,
            "recommendations": preference_result.get("filtered_recommendations", []),
        }


class HumanInTheLoop:
    """Human-in-the-loop approval system"""

    def __init__(self):
        self.pending_approvals = []

    def request_approval(self, recommendation):
        """Request human approval for recommendation"""
        approval_request = {
            "id": len(self.pending_approvals) + 1,
            "recommendation": recommendation,
            "status": "pending",
            "timestamp": "2024-01-01T00:00:00Z",  # Mock timestamp
        }
        self.pending_approvals.append(approval_request)
        return approval_request["id"]

    def process_approval(self, approval_id, decision, modifications=None):
        """Process human approval decision"""
        for request in self.pending_approvals:
            if request["id"] == approval_id:
                request["status"] = "approved" if decision else "rejected"
                if modifications:
                    request["modifications"] = modifications
                return request
        return None

    def get_approved_recommendations(self):
        """Get all approved recommendations"""
        return [req for req in self.pending_approvals if req["status"] == "approved"]


class Agent2:
    """Agent 2 - Context Engineering & Knowledge Base with LangChain-powered Thinking Layer"""

    def __init__(self):
        self.name = "Agent 2 - Context Engineering & Knowledge Base (AI-Powered)"
        self.state = "idle"
        self.context_store = ContextStore()
        self.knowledge_base = {}
        self.thinking_layer = (
            AgentThinkingLayer()
        )  # Now uses LangChain-powered thinking
        self.llm_service = (
            self.thinking_layer.llm_service
        )  # Add reference to LLM service
        self.human_loop = HumanInTheLoop()
        self._initialize_context_store()

    def process_campaign_request(
        self, campaign_data: Dict, customer_data: Dict, user_query: str = ""
    ) -> Dict[str, Any]:
        """Process campaign request through the LangChain-powered thinking layer with contextual memory"""
        print(f"\n{self.name}: Processing campaign request with AI...")
        print(f"Session ID: {self.thinking_layer.session_id}")

        # Get contextual information from memory and retrieval systems
        contextual_info = self._get_contextual_information(
            campaign_data, customer_data, user_query
        )

        # Prepare enhanced context for thinking layer
        context = self._prepare_ai_context(
            campaign_data, customer_data, contextual_info
        )

        # Process through AI-powered thinking layer
        thinking_result = self.thinking_layer.process(
            campaign_data, customer_data, context
        )

        # Store results in context store
        campaign_id = getattr(campaign_data, "messaging",
                              "unknown").replace(" ", "_")
        self._store_ai_context(campaign_id, thinking_result)

        # Generate final recommendation using AI insights with contextual enhancement
        recommendation = self._generate_contextual_recommendation(
            thinking_result,
            [customer_data] if not isinstance(
                customer_data, list) else customer_data,
            contextual_info,
        )

        # Store interaction in memory
        self._store_interaction(user_query, recommendation)

        return recommendation

    def _generate_contextual_recommendation(
        self, thinking_result, customer_data, contextual_info
    ):
        """Generate contextual recommendation using AI insights and memory"""
        # Start with base AI recommendation
        base_recommendation = self._generate_ai_recommendation(
            thinking_result, customer_data
        )

        # Enhance with contextual information
        if contextual_info:
            # Add contextual insights to the recommendation
            base_recommendation["contextual_enhancements"] = {
                "memory_insights": len(contextual_info.get("relevant_knowledge", [])),
                "campaign_history": len(contextual_info.get("campaign_insights", [])),
                "customer_history": len(contextual_info.get("customer_insights", [])),
                "conversation_context": bool(
                    contextual_info.get("conversation_history")
                ),
            }

            # Adjust recommendations based on historical data
            if contextual_info.get("campaign_insights"):
                base_recommendation["historical_learnings"] = [
                    "Applied insights from similar past campaigns",
                    "Optimized based on historical performance data",
                ]

        return base_recommendation

    def _store_interaction(self, user_query, recommendation):
        """Store interaction in memory system"""
        if (
            hasattr(self.thinking_layer, "memory_system")
            and self.thinking_layer.memory_system
        ):
            interaction_data = {
                "query": user_query,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.thinking_layer.session_id,
            }

            # Store the interaction
            self.thinking_layer.memory_system.store_interaction(
                user_query,
                json.dumps(recommendation, default=str),
                self.thinking_layer.session_id,
            )

            print(f"{self.name}: Stored interaction in memory system")

        print(f"{self.name}: AI-powered campaign processing completed")
        return recommendation

    def _store_ai_context(self, campaign_id: str, thinking_result):
        """Store AI thinking results in context store"""
        self.context_store.past_campaigns.append(
            {
                "campaign_id": campaign_id,
                "timestamp": datetime.now().isoformat(),
                "ai_thinking_result": thinking_result,
                "status": "ai_processed",
            }
        )

    def process(self):
        """Main processing logic for Agent 2 - AI-Powered Campaign Workflow"""
        print(f"\n{self.name}: Starting AI-powered campaign management workflow...")

        # Step 1: Gather campaign and customer data
        campaign_data = self._gather_campaign_data()
        customer_data = self._gather_customer_data()

        # Display sample data information
        print(f"\n📊 Campaign: {campaign_data.messaging}")
        print(f"💰 Budget: ${campaign_data.budget:,}")
        print(f"📢 Channels: {', '.join(campaign_data.channels)}")
        print(f"👥 Analyzing {len(customer_data)} customer profiles with AI")

        # Show customer segments
        all_segments = set()
        for customer in customer_data:
            all_segments.update(customer.segments)
        print(f"🎯 Target Segments: {', '.join(sorted(all_segments))}")

        # Step 2: Context engineering & knowledge base with AI enhancement
        context = self._prepare_ai_context(campaign_data, customer_data)
        print(
            f"📚 Using AI-enhanced context from {len(context.get('similar_campaigns', []))} past campaigns"
        )

        # Step 3: LangChain-powered Agent thinking layer processing
        print(f"🤖 Activating LangChain-powered AI thinking layer...")
        primary_customer = (
            customer_data[0] if customer_data else self._get_default_customer()
        )
        thinking_result = self.thinking_layer.process(
            campaign_data, primary_customer, context
        )

        # Step 4: Generate AI-powered recommendation
        recommendation = self._generate_ai_recommendation(
            thinking_result, customer_data
        )

        # Step 5: Human-in-the-loop approval with AI insights
        approval_id = self.human_loop.request_approval(recommendation)

        # Simulate human approval (in real system, this would be async)
        self._simulate_human_approval(approval_id)

        # Step 6: Process approved recommendations and update knowledge base
        approved_recs = self.human_loop.get_approved_recommendations()
        self._update_knowledge_base_with_ai_insights(thinking_result)

        print(
            f"\n✅ {self.name}: AI-powered campaign workflow completed successfully!")
        print(f"📋 Generated {len(approved_recs)} AI-approved recommendations")
        print(
            f"🎯 Targeting {len(customer_data)} customer profiles across {len(all_segments)} segments"
        )
        print(
            f"🤖 AI confidence score: {thinking_result.get('simulation_confidence', 0.8):.2f}"
        )

        return True

    def _gather_campaign_data(self):
        """Gather campaign data from sample data"""
        try:
            from sample_data import get_sample_campaign_data

            campaigns = get_sample_campaign_data()
            # Use the first campaign as default, or could be selected based on criteria
            return campaigns[0] if campaigns else self._get_default_campaign()
        except ImportError:
            return self._get_default_campaign()

    def _gather_customer_data(self):
        """Gather customer data from sample data"""
        try:
            from sample_data import get_sample_customer_data

            customers = get_sample_customer_data()
            # Return all customers for comprehensive analysis
            return customers
        except ImportError:
            return [self._get_default_customer()]

    def _initialize_context_store(self):
        """Initialize context store with sample historical data"""
        try:
            from sample_data import get_sample_context_data

            context_data = get_sample_context_data()

            # Load past campaigns
            for campaign_data in context_data["past_campaigns"]:
                self.context_store.past_campaigns.append(campaign_data)

            # Load behavior patterns
            self.context_store.behavior_patterns = context_data["behavior_patterns"]

            print(
                f"{self.name}: Loaded {len(context_data['past_campaigns'])} past campaigns into context store"
            )
        except ImportError:
            print(
                f"{self.name}: Sample data not available, using empty context store")

    def _get_default_campaign(self):
        """Get default campaign data using LLM if sample data unavailable"""
        try:
            # Create prompt for LLM to generate default campaign
            prompt = """
            Generate a default marketing campaign structure in JSON format:
            {
                "goals": ["goal1", "goal2"],
                "budget": 50000,
                "channels": ["channel1", "channel2", "channel3"],
                "messaging": "campaign message description",
                "timing": "campaign timing"
            }
            
            Create a balanced, realistic campaign suitable for most businesses.
            """

            response = self.llm_service.generate(prompt)
            campaign_data = self._parse_json_response(
                response,
                {
                    "goals": ["increase_awareness", "drive_sales"],
                    "budget": 50000,
                    "channels": ["email", "social_media", "display_ads"],
                    "messaging": "New product launch campaign",
                    "timing": "Q1 2024",
                },
            )

            campaign = CampaignData()
            campaign.goals = campaign_data.get(
                "goals", ["increase_awareness", "drive_sales"]
            )
            campaign.budget = campaign_data.get("budget", 50000)
            campaign.channels = campaign_data.get(
                "channels", ["email", "social_media", "display_ads"]
            )
            campaign.messaging = campaign_data.get(
                "messaging", "New product launch campaign"
            )
            campaign.timing = campaign_data.get("timing", "Q1 2024")
            return campaign

        except Exception as e:
            print(f"⚠️ Error generating LLM default campaign: {e}")
            # Fallback to static default
            campaign = CampaignData()
            campaign.goals = ["increase_awareness", "drive_sales"]
            campaign.budget = 50000
            campaign.channels = ["email", "social_media", "display_ads"]
            campaign.messaging = "New product launch campaign"
            campaign.timing = "Q1 2024"
            return campaign

    def _get_default_customer(self):
        """Get default customer data using LLM if sample data unavailable"""
        try:
            # Create prompt for LLM to generate default customer
            prompt = """
            Generate a default customer profile structure in JSON format:
            {
                "segments": ["segment1", "segment2"],
                "behavior": {
                    "preferred_channel": "channel_name",
                    "engagement_time": "time_period"
                },
                "interactions": ["interaction1", "interaction2", "interaction3"]
            }
            
            Create a realistic customer profile suitable for most businesses.
            """

            response = self.llm_service.generate(prompt)
            customer_data = self._parse_json_response(
                response,
                {
                    "segments": ["young_professionals", "tech_enthusiasts"],
                    "behavior": {
                        "preferred_channel": "email",
                        "engagement_time": "evening",
                    },
                    "interactions": ["website_visit", "email_open", "social_follow"],
                },
            )

            customer = CustomerData()
            customer.segments = customer_data.get(
                "segments", ["young_professionals", "tech_enthusiasts"]
            )
            customer.behavior = customer_data.get(
                "behavior", {"preferred_channel": "email",
                             "engagement_time": "evening"}
            )
            customer.interactions = customer_data.get(
                "interactions", ["website_visit",
                                 "email_open", "social_follow"]
            )
            return customer

        except Exception as e:
            print(f"⚠️ Error generating LLM default customer: {e}")
            # Fallback to static default
            customer = CustomerData()
            customer.segments = ["young_professionals", "tech_enthusiasts"]
            customer.behavior = {
                "preferred_channel": "email",
                "engagement_time": "evening",
            }
            customer.interactions = [
                "website_visit", "email_open", "social_follow"]
            return customer

    def _generate_ai_recommendation(self, thinking_result, customer_data):
        """Generate AI-powered recommendation based on LangChain thinking result and customer data"""
        # Extract AI insights from thinking result
        ai_insights = thinking_result.get("ai_insights", {})

        # Analyze customer preferences for channel optimization with AI enhancement
        channel_preferences = {}
        total_value = 0

        for customer in customer_data:
            pref_channel = customer.behavior.get("preferred_channel", "email")
            avg_value = customer.behavior.get("avg_order_value", 100)

            if pref_channel not in channel_preferences:
                channel_preferences[pref_channel] = {
                    "count": 0, "total_value": 0}

            channel_preferences[pref_channel]["count"] += 1
            channel_preferences[pref_channel]["total_value"] += avg_value
            total_value += avg_value

        # Calculate AI-optimized budget allocation
        budget_allocation = {}
        base_budget = 50000  # Default budget

        # Use AI preferences if available
        ai_channel_prefs = ai_insights.get("preferences", {}).get(
            "channel_preferences", {}
        )
        if ai_channel_prefs:
            for channel, ai_weight in ai_channel_prefs.items():
                budget_allocation[channel] = int(base_budget * ai_weight)
        else:
            # Fallback to customer data analysis
            for channel, data in channel_preferences.items():
                weight = (
                    (data["total_value"] /
                     total_value) if total_value > 0 else 0.25
                )
                budget_allocation[channel] = int(base_budget * weight)

        # Generate customer segment insights
        segments = set()
        for customer in customer_data:
            segments.update(customer.segments)

        return {
            "campaign_strategy": f"AI-optimized multi-channel approach targeting {len(segments)} segments",
            "budget_allocation": budget_allocation,
            "predicted_outcomes": thinking_result,
            "ai_confidence": thinking_result.get("simulation_confidence", 0.8),
            "timeline": "4-6 weeks (AI-optimized)",
            "success_metrics": [
                f"ROI > {thinking_result.get('predicted_roi', 2.5)}",
                f"Reach > {thinking_result.get('expected_reach', len(customer_data) * 1000)}",
                f"Conversion > {thinking_result.get('conversion_rate', 0.05) * 100:.1f}%",
            ],
            "target_segments": list(segments),
            "ai_recommendations": thinking_result.get("recommendations", []),
            "customer_insights": {
                "total_customers": len(customer_data),
                "avg_order_value": (
                    total_value / len(customer_data) if customer_data else 0
                ),
                "preferred_channels": channel_preferences,
                "ai_enhanced": True,
            },
            "ai_insights": ai_insights,
        }

    def _parse_json_response(self, response: str, fallback: Any) -> Any:
        """Safely parse JSON response with fallback"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except (json.JSONDecodeError, AttributeError):
            print(f"⚠️ Failed to parse JSON response, using fallback")
            return fallback

    def _simulate_human_approval(self, approval_id):
        """Simulate human approval process with AI insights"""
        # In real system, this would be handled by actual human input
        print(
            f"{self.name}: Requesting human approval for AI-generated recommendation {approval_id}..."
        )

        # Simulate approval with AI-informed modifications
        modifications = {
            "budget_adjustment": "AI suggests increasing social media budget by 10%",
            "timeline_change": "AI recommends extending to 5 weeks for optimal results",
            "ai_monitoring": "Enable continuous AI optimization during campaign",
        }

        self.human_loop.process_approval(approval_id, True, modifications)
        print(
            f"{self.name}: AI recommendation {approval_id} approved with AI-informed modifications"
        )

    def communicate_with_agent1(self, message):
        """Send message to Agent 1"""
        print(f"{self.name}: Sending message to Agent1: {message}")
        return f"Message sent to Agent1: {message}"

    def use_tools(self, tool_name, *args, **kwargs):
        """Access shared tools"""
        print(f"{self.name}: Using tool {tool_name} with args {args}")
        return f"Tool {tool_name} executed successfully"

    def continue_process(self):
        """Continue processing after tool usage"""
        print(f"{self.name}: Continuing campaign workflow process...")
        return self.process()

    def run_campaign_workflow(self):
        """Run the complete AI-powered campaign workflow"""
        print(f"\n=== {self.name} AI-Powered Campaign Management Workflow ===")
        print("AI-Enhanced Workflow Components:")
        print("1. Campaign Data Input (goals, budget, channels, messaging, timing)")
        print("2. Customer Data Input (segments, behavior, interactions)")
        print("3. AI-Enhanced Context Engineering & Knowledge Base")
        print("4. LangChain-Powered Agent Thinking Layer:")
        print("   - AI Analysis Node (LangChain + OpenAI)")
        print("   - AI Reasoning Node (Strategic AI reasoning)")
        print("   - AI Preference Node (AI preference analysis)")
        print("   - AI Simulation Node (AI outcome prediction)")
        print("5. AI-Generated Recommendation")
        print("6. Human-in-the-Loop Approval (with AI insights)")
        print("7. AI-Optimized Campaign Execution Plan")
        print("8. Continuous AI Learning & Knowledge Base Updates")
        print("\n🤖 Starting AI-powered workflow execution...")

        return self.process()

    def _get_contextual_information(
        self, campaign_data: Dict[str, Any], customer_data: Any, user_query: str
    ) -> Dict[str, Any]:
        """Get contextual information from memory and retrieval systems"""
        contextual_info = {
            "conversation_history": "",
            "relevant_knowledge": [],
            "campaign_insights": [],
            "customer_insights": [],
            "timestamp": datetime.now().isoformat(),
        }

        if (
            hasattr(self.thinking_layer, "memory_system")
            and self.thinking_layer.memory_system
        ):
            # Get conversation context
            contextual_info["conversation_history"] = (
                self.thinking_layer.memory_system.get_conversation_context()
            )

            # Get general contextual information
            if user_query:
                general_context = (
                    self.thinking_layer.memory_system.get_contextual_information(
                        user_query
                    )
                )
                contextual_info["relevant_knowledge"] = general_context.get(
                    "relevant_knowledge", []
                )

        if (
            hasattr(self.thinking_layer, "retrieval_system")
            and self.thinking_layer.retrieval_system
        ):
            # Get campaign-specific insights
            campaign_insights = (
                self.thinking_layer.retrieval_system.retrieve_campaign_insights(
                    campaign_data
                )
            )
            contextual_info["campaign_insights"] = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in campaign_insights
            ]

            # Get customer-specific insights
            customer_insights = (
                self.thinking_layer.retrieval_system.retrieve_customer_insights(
                    customer_data
                )
            )
            contextual_info["customer_insights"] = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in customer_insights
            ]

        return contextual_info

    def _prepare_ai_context(
        self, campaign_data, customer_data, contextual_info: Dict[str, Any] = None
    ):
        """Prepare AI-enhanced context for thinking layer processing with memory integration"""
        base_context = self.context_store.get_relevant_context(campaign_data)

        # Add AI-specific context enhancements
        ai_context = {
            **base_context,
            "timestamp": datetime.now().isoformat(),
            "market_conditions": {
                "competition_level": "moderate",
                "ai_adoption_rate": "high",
                "personalization_demand": "increasing",
            },
            "customer_insights": {
                "total_profiles": (
                    1 if not isinstance(
                        customer_data, list) else len(customer_data)
                ),
                "ai_personalization_readiness": "high",
                "behavioral_patterns": (
                    [customer_data.behavior]
                    if not isinstance(customer_data, list)
                    else [customer.behavior for customer in customer_data]
                ),
            },
            "ai_capabilities": {
                "langchain_integration": True,
                "openai_powered": True,
                "continuous_learning": True,
            },
        }

        # Add contextual information if available
        if contextual_info:
            ai_context["contextual_memory"] = {
                "conversation_history": contextual_info.get("conversation_history", ""),
                "relevant_knowledge_count": len(
                    contextual_info.get("relevant_knowledge", [])
                ),
                "campaign_insights_count": len(
                    contextual_info.get("campaign_insights", [])
                ),
                "customer_insights_count": len(
                    contextual_info.get("customer_insights", [])
                ),
            }

            # Add specific insights to context
            if contextual_info.get("campaign_insights"):
                ai_context["retrieved_campaign_insights"] = [
                    insight["content"]
                    for insight in contextual_info["campaign_insights"][:3]
                ]

            if contextual_info.get("customer_insights"):
                ai_context["retrieved_customer_insights"] = [
                    insight["content"]
                    for insight in contextual_info["customer_insights"][:3]
                ]

        return ai_context

    def _update_knowledge_base_with_ai_insights(self, thinking_result):
        """Update knowledge base with AI-generated insights"""
        ai_insights = thinking_result.get("ai_insights", {})

        # Store AI insights in knowledge base
        insight_key = f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.knowledge_base[insight_key] = {
            "timestamp": datetime.now().isoformat(),
            "ai_analysis": ai_insights.get("analysis", {}),
            "ai_reasoning": ai_insights.get("reasoning", {}),
            "ai_preferences": ai_insights.get("preferences", {}),
            "ai_simulation": ai_insights.get("simulation", {}),
            "confidence_score": thinking_result.get("simulation_confidence", 0.8),
            "recommendations": thinking_result.get("recommendations", []),
        }

        print(f"{self.name}: Updated knowledge base with AI insights ({insight_key})")

    def get_ai_knowledge_summary(self):
        """Get summary of AI-generated knowledge"""
        ai_entries = [
            k for k in self.knowledge_base.keys() if k.startswith("ai_insights_")
        ]

        return {
            "total_ai_insights": len(ai_entries),
            "knowledge_base_size": len(self.knowledge_base),
            "ai_learning_active": True,
            "langchain_integration": "active",
            "last_ai_update": max(
                [self.knowledge_base[k].get("timestamp", "")
                 for k in ai_entries],
                default="never",
            ),
        }

    def process_agent1_output(
        self, optimized_campaigns: List[Dict[str, Any]], customer_data: Dict
    ) -> Dict[str, Any]:
        """Process optimized output from Agent1, enhancing thinking and decisioning."""
        context = self._prepare_ai_context(
            optimized_campaigns, customer_data, {})
        thinking_result = self.thinking_layer.process(
            optimized_campaigns, customer_data, context
        )
        enhanced_decision = self._enhance_decision(
            thinking_result, optimized_campaigns)
        return enhanced_decision

    def run_implementation_workflow_with_status(self, agent1_output, socketio):
        """Run the implementation workflow with real-time status updates via SocketIO"""
        try:
            # Starting implementation
            socketio.emit(
                "agent2_status",
                {
                    "status": "starting",
                    "message": "Starting Agent 2 implementation workflow...",
                    "step": "initialization",
                },
            )
            time.sleep(1)

            # Processing Agent 1 output
            socketio.emit(
                "agent2_status",
                {
                    "status": "thinking",
                    "message": "Processing Agent 1 output and analyzing campaigns...",
                    "step": "analysis",
                },
            )
            time.sleep(2)

            # Creating implementation plan
            socketio.emit(
                "agent2_status",
                {
                    "status": "progress",
                    "message": "Creating detailed implementation plan...",
                    "step": "planning",
                },
            )

            # Generate implementation plan
            implementation_plan = self._generate_implementation_plan(
                agent1_output)

            # Emit implementation data with complete results
            complete_result = {
                "agent1_result": agent1_output,
                "agent2_result": implementation_plan,
                "implementation": implementation_plan,  # Keep for backward compatibility
            }
            socketio.emit("implementation_ready", complete_result)
            socketio.emit(
                "agent2_status",
                {
                    "status": "progress",
                    "message": "Implementation plan created successfully",
                    "step": "planning",
                    "data": complete_result,
                },
            )
            time.sleep(2)

            # Resource allocation
            socketio.emit(
                "agent2_status",
                {
                    "status": "progress",
                    "message": "Allocating resources and setting up infrastructure...",
                    "step": "resource_allocation",
                },
            )
            time.sleep(2)

            # Final optimization
            socketio.emit(
                "agent2_status",
                {
                    "status": "progress",
                    "message": "Optimizing implementation strategy...",
                    "step": "optimization",
                },
            )
            time.sleep(1)

            # Completed
            socketio.emit(
                "agent2_status",
                {
                    "status": "completed",
                    "message": "Agent 2 implementation workflow completed successfully!",
                    "step": "completed",
                },
            )

            return {"implementation_plan": implementation_plan, "status": "completed"}

        except Exception as e:
            socketio.emit(
                "agent2_status",
                {
                    "status": "error",
                    "message": f"Error in Agent 2 workflow: {str(e)}",
                    "step": "error",
                },
            )
            return {"status": "error", "message": str(e)}

    def _generate_implementation_plan(self, agent1_output):
        """Generate comprehensive campaign analysis with detailed action recommendations, scenarios, timing, and reasoning"""
        campaigns = agent1_output.get(
            "marketing_campaigns", agent1_output.get("campaigns", [])
        )
        segments = agent1_output.get(
            "customer_segments", agent1_output.get("segments", [])
        )
        goals = agent1_output.get(
            "business_goals", agent1_output.get("goals", []))

        print(
            f"\n🔍 Agent2: Processing {len(campaigns)} campaigns, {len(segments)} segments, {len(goals)} goals from Agent1"
        )

        # Analyze each campaign from Agent 1
        campaign_analysis = self._analyze_campaigns_detailed(
            campaigns, segments)

        # Generate decision flows based on actual Agent1 output
        decision_flows = self._generate_decision_flows_from_agent1_data(
            campaigns, segments, goals
        )

        print(
            f"✅ Agent2: Generated {len(decision_flows)} personalized decision flows based on Agent1 output"
        )

        # Generate AI-powered execution strategy using LLM
        ai_execution_strategy = self._generate_ai_execution_strategy(
            campaigns, segments, goals
        )

        # Generate campaign performance predictions
        campaign_predictions = self._generate_campaign_predictions(
            campaigns, segments, goals
        )

        # Identify top-performing campaigns
        top_campaigns = self._identify_top_performing_campaigns(
            campaign_predictions)

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            campaigns, campaign_predictions
        )

        # Create performance summary for final results
        performance_summary = {
            "total_predicted_conversions": sum(
                pred["expected_metrics"]["total_conversions"]
                for pred in campaign_predictions.values()
            ),
            "average_conversion_rate": (
                round(
                    sum(
                        pred["expected_metrics"]["conversion_rate"]
                        for pred in campaign_predictions.values()
                    )
                    / len(campaign_predictions),
                    2,
                )
                if campaign_predictions
                else 0
            ),
            "average_ctr": (
                round(
                    sum(
                        pred["expected_metrics"]["ctr"]
                        for pred in campaign_predictions.values()
                    )
                    / len(campaign_predictions),
                    2,
                )
                if campaign_predictions
                else 0
            ),
            "total_expected_revenue": sum(
                pred["expected_metrics"]["estimated_revenue"]
                for pred in campaign_predictions.values()
            ),
            "average_engagement_score": (
                round(
                    sum(
                        pred["expected_metrics"]["customer_engagement_score"]
                        for pred in campaign_predictions.values()
                    )
                    / len(campaign_predictions),
                    1,
                )
                if campaign_predictions
                else 0
            ),
        }

        return {
            "decision_flows": decision_flows,
            "ai_execution_strategy": ai_execution_strategy,
            "performance_tracking": self._generate_performance_tracking_strategy(
                campaigns, segments
            ),
            "campaign_analysis": (
                campaign_analysis if "campaign_analysis" in locals() else {}
            ),
            "campaign_predictions": campaign_predictions,
            "top_performing_campaigns": top_campaigns,
            "improvement_suggestions": improvement_suggestions,
            "performance_summary": performance_summary,
        }

    def _generate_ai_execution_strategy(self, campaigns, segments, goals):
        """Generate AI-powered execution strategy using LLM"""
        try:
            # Create detailed prompt for LLM to generate execution strategy
            prompt = f"""
            Generate an AI-powered execution strategy for marketing campaigns:
            
            Campaign Context:
            - Number of campaigns: {len(campaigns)}
            - Customer segments: {len(segments)}
            - Business goals: {goals}
            
            Generate execution strategy in JSON format:
            {{
                "real_time_decisioning": "description of real-time decision capabilities",
                "personalization_engine": "description of personalization approach",
                "value_prediction": "description of predictive analytics",
                "learning_system": "description of continuous learning approach"
            }}
            
            Focus on:
            - Real-time behavioral triggers and responses
            - Personalization at scale for multiple segments
            - Predictive analytics for customer value
            - Continuous learning and optimization
            """

            response = self.llm_service.generate(prompt)
            strategy_data = self._parse_json_response(
                response,
                {
                    "real_time_decisioning": "Advanced behavioral trigger system",
                    "personalization_engine": f"Dynamic profiles for {len(segments)} segments",
                    "value_prediction": "ML-driven customer value prediction",
                    "learning_system": "Continuous optimization system",
                },
            )
            return strategy_data

        except Exception as e:
            print(f"⚠️ Error generating LLM execution strategy: {e}")
            # Fallback to dynamic default
            return {
                "real_time_decisioning": f"Behavioral trigger system for {len(campaigns)} campaigns",
                "personalization_engine": f"Dynamic profiles for {len(segments)} customer segments",
                "value_prediction": "ML-driven customer lifetime value prediction",
                "learning_system": "Continuous learning from campaign performance",
            }

    def _generate_performance_tracking_strategy(self, campaigns, segments):
        """Generate performance tracking strategy using LLM"""
        try:
            # Create detailed prompt for LLM to generate tracking strategy
            prompt = f"""
            Generate a performance tracking strategy for marketing campaigns:
            
            Campaign Context:
            - Number of campaigns: {len(campaigns)}
            - Customer segments: {len(segments)}
            
            Generate tracking strategy in JSON format:
            {{
                "real_time_metrics": [
                    "metric 1",
                    "metric 2",
                    "metric 3",
                    "metric 4"
                ],
                "ab_testing": "description of A/B testing approach",
                "optimization": "description of optimization strategy"
            }}
            
            Focus on:
            - Real-time performance metrics
            - A/B testing methodologies
            - Continuous optimization approaches
            """

            response = self.llm_service.generate(prompt)
            tracking_data = self._parse_json_response(
                response,
                {
                    "real_time_metrics": [
                        "Click-through rates by segment",
                        "Conversion rates by channel",
                        "Revenue attribution by campaign",
                        "Customer lifetime value impact",
                    ],
                    "ab_testing": "Automated testing of message variants and timing",
                    "optimization": "Continuous learning from user responses",
                },
            )
            return tracking_data

        except Exception as e:
            print(f"⚠️ Error generating LLM tracking strategy: {e}")
            # Fallback to dynamic default
            return {
                "real_time_metrics": [
                    f"Performance metrics for {len(campaigns)} campaigns",
                    f"Segment analysis for {len(segments)} groups",
                    "Revenue attribution tracking",
                    "Customer engagement scoring",
                ],
                "ab_testing": "Automated testing of campaign variants",
                "optimization": "Continuous performance optimization",
            }

    def _analyze_campaigns_detailed(self, campaigns, segments):
        """Analyze each campaign in detail with specific recommendations"""
        analysis = {}

        for i, campaign in enumerate(campaigns):
            campaign_id = f"campaign_{i+1}"

            # Extract campaign details
            campaign_type = campaign.get("type", "general")
            target_segment = campaign.get("target_segment", "all_customers")
            message = campaign.get("message", "")
            channel = campaign.get("channel", "email")
            budget = campaign.get("budget", 0)

            # Analyze campaign effectiveness potential
            effectiveness_score = self._calculate_effectiveness_score(
                campaign, segments
            )

            # Generate specific action recommendations
            action_recommendations = self._generate_action_recommendations(
                campaign, target_segment
            )

            # Determine optimal timing
            timing_strategy = self._determine_optimal_timing(
                campaign, target_segment)

            # Identify scenarios and triggers
            scenarios = self._identify_campaign_scenarios(
                campaign, target_segment)

            analysis[campaign_id] = {
                "campaign_details": {
                    "type": campaign_type,
                    "target_segment": target_segment,
                    "message": message,
                    "channel": channel,
                    "budget": budget,
                },
                "effectiveness_analysis": {
                    "score": effectiveness_score,
                    "reasoning": self._explain_effectiveness_reasoning(
                        campaign, effectiveness_score
                    ),
                    "improvement_suggestions": self._suggest_improvements(campaign),
                },
                "action_recommendations": action_recommendations,
                "timing_strategy": timing_strategy,
                "scenarios": scenarios,
                "success_metrics": self._define_success_metrics(campaign),
                "risk_assessment": self._assess_campaign_risks(campaign),
            }

        return analysis

    def _generate_decision_flows_from_agent1_data(self, campaigns, segments, goals):
        """Generate personalized decision flows based on actual Agent1 output"""
        flows = {}

        # Create flows for each segment from Agent1
        for i, segment in enumerate(segments):
            segment_name = (
                segment.get("name", f"segment_{i+1}").lower().replace(" ", "_")
            )
            segment_description = segment.get(
                "description", "Customer segment")
            # Remove size reference - focus on segment criteria and characteristics
            segment_characteristics = segment.get("characteristics", {})

            # Find relevant campaigns for this segment
            relevant_campaigns = []
            for campaign in campaigns:
                campaign_targets = campaign.get("target_segments", [])
                if (
                    segment_name in campaign_targets
                    or segment.get("name") in campaign_targets
                ):
                    relevant_campaigns.append(campaign)

            # Find relevant goals for this segment
            relevant_goals = []
            for goal in goals:
                goal_targets = goal.get("target_segments", [])
                if segment_name in goal_targets or segment.get("name") in goal_targets:
                    relevant_goals.append(goal)

            # Generate actions based on campaigns
            actions = []
            channels = set()
            for campaign in relevant_campaigns:
                campaign_channels = campaign.get("channels", [])
                channels.update(campaign_channels)

                # Extract campaign content for actions
                content = campaign.get("content", {})
                for channel, message in content.items():
                    actions.append(
                        f"Deploy {channel} campaign: {message[:100]}..."
                        if len(message) > 100
                        else f"Deploy {channel} campaign: {message}"
                    )

            # Generate context factors based on segment characteristics
            context_factors = []
            if isinstance(segment_characteristics, dict):
                for key, value in segment_characteristics.items():
                    context_factors.append(f"{key}: {value}")
            elif isinstance(segment_characteristics, list):
                context_factors.extend(segment_characteristics)

            # Add default context factors if none exist
            if not context_factors:
                context_factors = [
                    "Customer engagement history and preferences",
                    "Previous campaign response patterns",
                    "Optimal timing based on segment behavior",
                    "Channel effectiveness for this segment",
                ]

            # Generate success metrics based on goals
            success_metrics = []
            for goal in relevant_goals:
                goal_metrics = goal.get("success_metrics", [])
                success_metrics.extend(goal_metrics)

            if not success_metrics:
                success_metrics = [
                    "Conversion rate improvement",
                    "Customer engagement increase",
                    "ROI optimization",
                    "Customer lifetime value growth",
                ]

            flows[segment_name] = {
                "goal": f"Optimize marketing effectiveness for {segment.get('name', segment_name)}",
                "criteria": segment_description,
                "actions": (
                    actions
                    if actions
                    else [
                        f"Develop targeted campaign for {segment.get('name', segment_name)}"
                    ]
                ),
                "context_factors": context_factors,
                "success_metrics": success_metrics,
                "relevant_campaigns": len(relevant_campaigns),
                "relevant_goals": len(relevant_goals),
                "channels": list(channels) if channels else ["email", "social_media"],
                "priority_score": segment.get("priority_score"),
                "characteristics": segment_characteristics,
            }

        return flows

    def _calculate_effectiveness_score(self, campaign, segments):
        """Calculate campaign effectiveness score based on multiple factors"""
        score = 0.5  # Base score

        # Channel effectiveness
        channel = campaign.get("channel", "email")
        channel_scores = {"email": 0.7, "sms": 0.8, "push": 0.6, "social": 0.5}
        score += channel_scores.get(channel, 0.5) * 0.3

        # Message personalization
        message = campaign.get("message", "")
        if "personalized" in message.lower() or "{name}" in message:
            score += 0.2

        # Budget adequacy
        budget = campaign.get("budget", 0)
        if budget > 1000:
            score += 0.1
        elif budget > 500:
            score += 0.05

        return min(score, 1.0)

    def _generate_action_recommendations(self, campaign, target_segment):
        """Generate specific action recommendations for the campaign using LLM"""
        try:
            # Create detailed prompt for LLM to generate action recommendations
            prompt = f"""
            Generate specific, actionable recommendations for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Budget: {campaign.get('budget', 'Not specified')}
            - Message: {campaign.get('message', 'Not specified')}
            - Goals: {campaign.get('goals', [])}
            
            Target Segment: {target_segment}
            
            Generate 3-5 specific action recommendations in JSON format:
            {{
                "recommendations": [
                    {{
                        "action": "specific_action_name",
                        "description": "detailed description of what to do",
                        "reasoning": "why this action will be effective",
                        "priority": "high/medium/low",
                        "expected_impact": "quantified expected improvement"
                    }}
                ]
            }}
            
            Focus on:
            - Channel-specific optimizations
            - Segment-specific personalization
            - Performance improvement tactics
            - Timing and frequency optimization
            - Content and messaging enhancements
            """

            response = self.llm_service.generate(prompt)
            recommendations_data = self._parse_json_response(
                response, {"recommendations": []}
            )
            return recommendations_data.get("recommendations", [])

        except Exception as e:
            print(f"⚠️ Error generating LLM action recommendations: {e}")
            # Fallback to basic LLM-generated recommendations
            fallback_prompt = f"""
            Generate 2-3 basic action recommendations for a {campaign.get('channel', 'email')} campaign 
            targeting {target_segment}. Return as JSON array of objects with action, description, and priority fields.
            """
            try:
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, [])
            except:
                # Final fallback - minimal dynamic response
                return [
                    {
                        "action": "optimize_targeting",
                        "description": f"Refine targeting for {target_segment} segment",
                        "reasoning": "Better targeting improves campaign effectiveness",
                        "priority": "high",
                    }
                ]

    def _determine_optimal_timing(self, campaign, target_segment):
        """Determine optimal timing strategy for the campaign using LLM"""
        try:
            # Create detailed prompt for LLM to generate timing strategy
            prompt = f"""
            Analyze and determine the optimal timing strategy for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Target Segment: {target_segment}
            - Goals: {campaign.get('goals', [])}
            
            Generate optimal timing strategy in JSON format:
            {{
                "timing_strategy": {{
                    "optimal_days": ["list of best days"],
                    "optimal_hours": ["list of best times"],
                    "frequency": "recommended frequency",
                    "reasoning": "detailed explanation of timing choices",
                    "seasonal_considerations": "any seasonal factors",
                    "timezone_recommendations": "timezone considerations"
                }}
            }}
            
            Consider:
            - Target segment behavior patterns
            - Channel-specific engagement times
            - Industry best practices
            - Campaign type requirements
            - Geographic and demographic factors
            """

            response = self.llm_service.generate(prompt)
            timing_data = self._parse_json_response(
                response, {"timing_strategy": {}})
            return timing_data.get("timing_strategy", {})

        except Exception as e:
            print(f"⚠️ Error generating LLM timing strategy: {e}")
            # Fallback to basic LLM-generated timing
            fallback_prompt = f"""
            Generate basic timing recommendations for a {campaign.get('channel', 'email')} campaign 
            targeting {target_segment}. Include optimal days, hours, and frequency in JSON format.
            """
            try:
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, {})
            except:
                # Enhanced fallback with LLM-generated content
                try:
                    enhanced_fallback_prompt = f"""
                    Generate optimal timing recommendations for a {campaign.get('channel', 'email')} campaign 
                    targeting {target_segment}. Include optimal days, hours, frequency, and reasoning in JSON format.
                    Consider the target segment characteristics and campaign type.
                    """
                    enhanced_response = self.llm_service.generate(
                        enhanced_fallback_prompt
                    )
                    return self._parse_json_response(
                        enhanced_response,
                        {
                            "optimal_days": ["Tuesday", "Wednesday", "Thursday"],
                            "optimal_hours": ["10:00 AM", "2:00 PM"],
                            "frequency": "Weekly",
                            "reasoning": f"Standard timing optimized for {target_segment} segment",
                        },
                    )
                except:
                    # Final static fallback only if LLM completely fails
                    return {
                        "optimal_days": ["Tuesday", "Wednesday", "Thursday"],
                        "optimal_hours": ["10:00 AM", "2:00 PM"],
                        "frequency": "Weekly",
                        "reasoning": f"Standard timing optimized for {target_segment} segment",
                    }

        if campaign_type == "promotional":
            timing_strategy["frequency"] = "Event-based"
            timing_strategy[
                "reasoning"
            ] += " Promotional campaigns should align with shopping patterns and seasonal events"

        return timing_strategy

    def _identify_campaign_scenarios(self, campaign, target_segment):
        """Identify specific scenarios and triggers for the campaign using LLM"""
        try:
            # Create detailed prompt for LLM to generate campaign scenarios
            prompt = f"""
            Analyze and identify specific scenarios, triggers, and decision points for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Type: {campaign.get('type', 'general')}
            - Channel: {campaign.get('channel', 'email')}
            - Target Segment: {target_segment}
            - Goals: {campaign.get('goals', [])}
            - Message: {campaign.get('message', 'Not specified')}
            
            Generate comprehensive scenarios in JSON format:
            {{
                "scenarios": {{
                    "trigger_events": ["list of specific trigger events"],
                    "context_factors": ["list of contextual factors to consider"],
                    "decision_points": ["list of key decision points"],
                    "automation_rules": ["list of automation conditions"],
                    "personalization_triggers": ["list of personalization opportunities"]
                }}
            }}
            
            Consider:
            - Campaign type-specific triggers
            - Customer behavior patterns
            - Channel-specific considerations
            - Timing and frequency rules
            - Personalization opportunities
            - Performance optimization points
            """

            response = self.llm_service.generate(prompt)
            scenarios_data = self._parse_json_response(
                response, {"scenarios": {}})
            return scenarios_data.get("scenarios", {})

        except Exception as e:
            print(f"⚠️ Error generating LLM campaign scenarios: {e}")
            # Fallback to basic LLM-generated scenarios
            fallback_prompt = f"""
            Generate basic trigger events and decision points for a {campaign.get('type', 'general')} 
            campaign on {campaign.get('channel', 'email')} targeting {target_segment}. 
            Return as JSON with trigger_events, context_factors, and decision_points arrays.
            """
            try:
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, {})
            except:
                # Enhanced fallback with LLM-generated content
                try:
                    enhanced_fallback_prompt = f"""
                    Generate campaign scenarios for {campaign.get('type', 'general')} campaign targeting {target_segment}.
                    Return JSON with trigger_events, context_factors, and decision_points arrays.
                    Make it specific to the campaign type and target segment.
                    """
                    enhanced_response = self.llm_service.generate(
                        enhanced_fallback_prompt
                    )
                    return self._parse_json_response(
                        enhanced_response,
                        {
                            "trigger_events": [
                                f"User matches {target_segment} criteria",
                                "Optimal timing window reached",
                            ],
                            "context_factors": [
                                "User engagement history",
                                "Channel preferences",
                                "Timing considerations",
                            ],
                            "decision_points": [
                                "Should campaign be sent?",
                                "Which message variant to use?",
                                "What follow-up actions?",
                            ],
                        },
                    )
                except:
                    # Final static fallback only if LLM completely fails
                    return {
                        "trigger_events": [
                            f"User matches {target_segment} criteria",
                            "Optimal timing window reached",
                        ],
                        "context_factors": [
                            "User engagement history",
                            "Channel preferences",
                            "Timing considerations",
                        ],
                        "decision_points": [
                            "Should campaign be sent?",
                            "Which message variant to use?",
                            "What follow-up actions?",
                        ],
                    }

    def _explain_effectiveness_reasoning(self, campaign, score):
        """Explain the reasoning behind the effectiveness score using LLM"""
        try:
            # Create detailed prompt for LLM to generate effectiveness reasoning
            prompt = f"""
            Analyze and explain the effectiveness reasoning for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Budget: {campaign.get('budget', 'Not specified')}
            - Goals: {campaign.get('goals', [])}
            - Target Segments: {campaign.get('target_segments', [])}
            - Message: {campaign.get('message', 'Not specified')}
            
            Effectiveness Score: {score} (0.0 - 1.0 scale)
            
            Generate detailed reasoning in JSON format:
            {{
                "reasoning": [
                    "specific reason 1",
                    "specific reason 2",
                    "specific reason 3"
                ]
            }}
            
            Explain:
            - Why this score was assigned
            - Channel effectiveness factors
            - Budget impact on performance
            - Target audience alignment
            - Message and content quality
            - Timing and execution factors
            """

            response = self.llm_service.generate(prompt)
            reasoning_data = self._parse_json_response(
                response, {"reasoning": []})
            return reasoning_data.get("reasoning", [])

        except Exception as e:
            print(f"⚠️ Error generating LLM effectiveness reasoning: {e}")
            # Fallback to basic LLM-generated reasoning
            try:
                fallback_prompt = f"""
                Explain why a {campaign.get('channel', 'email')} campaign with score {score} 
                would have this effectiveness level. Return 2-3 reasons as JSON array.
                """
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, [])
            except:
                # Enhanced fallback with LLM-generated content
                try:
                    enhanced_fallback_prompt = f"""
                    Explain why a {campaign.get('channel', 'email')} marketing campaign 
                    named '{campaign.get('name', 'Unnamed Campaign')}' with effectiveness score {score} 
                    would have this performance level. Provide 2-3 specific reasons as JSON array.
                    """
                    enhanced_response = self.llm_service.generate(
                        enhanced_fallback_prompt
                    )
                    return self._parse_json_response(enhanced_response, [])
                except:
                    # Final static fallback only if LLM completely fails
                    if score >= 0.8:
                        return [
                            f"High effectiveness expected for {campaign.get('channel', 'email')} campaign",
                            "Strong targeting and execution",
                        ]
                    elif score >= 0.6:
                        return [
                            f"Moderate effectiveness for {campaign.get('channel', 'email')} campaign",
                            "Room for optimization exists",
                        ]
                    else:
                        return [
                            f"Lower effectiveness for {campaign.get('channel', 'email')} campaign",
                            "Significant improvements needed",
                        ]

    def _suggest_improvements(self, campaign):
        """Suggest specific improvements for the campaign using LLM"""
        try:
            # Create detailed prompt for LLM to generate improvement suggestions
            prompt = f"""
            Analyze this marketing campaign and suggest specific improvements:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Budget: {campaign.get('budget', 'Not specified')}
            - Message: {campaign.get('message', 'Not specified')}
            - Goals: {campaign.get('goals', [])}
            - Target Segments: {campaign.get('target_segments', [])}
            
            Generate specific improvement suggestions in JSON format:
            {{
                "improvements": [
                    "specific improvement suggestion 1",
                    "specific improvement suggestion 2",
                    "specific improvement suggestion 3"
                ]
            }}
            
            Focus on:
            - Message optimization (length, clarity, personalization)
            - Channel-specific best practices
            - Budget allocation improvements
            - Targeting and segmentation enhancements
            - Call-to-action optimization
            - A/B testing opportunities
            """

            response = self.llm_service.generate(prompt)
            improvements_data = self._parse_json_response(
                response, {"improvements": []}
            )
            return improvements_data.get("improvements", [])

        except Exception as e:
            print(f"⚠️ Error generating LLM improvement suggestions: {e}")
            # Fallback to basic LLM-generated improvements
            try:
                fallback_prompt = f"""
                Generate 3 basic improvement suggestions for a {campaign.get('channel', 'email')} 
                marketing campaign. Return as JSON array of strings.
                """
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, [])
            except:
                # Enhanced fallback with LLM-generated content
                try:
                    enhanced_fallback_prompt = f"""
                    Generate 3 specific improvement suggestions for a {campaign.get('channel', 'email')} 
                    marketing campaign named '{campaign.get('name', 'Unnamed Campaign')}'. 
                    Focus on actionable improvements. Return as JSON array of strings.
                    """
                    enhanced_response = self.llm_service.generate(
                        enhanced_fallback_prompt
                    )
                    return self._parse_json_response(
                        enhanced_response,
                        [
                            f"Optimize {campaign.get('channel', 'email')} campaign targeting",
                            "Improve message personalization and clarity",
                            "Test different timing and frequency strategies",
                        ],
                    )
                except:
                    # Final static fallback only if LLM completely fails
                    return [
                        f"Optimize {campaign.get('channel', 'email')} campaign targeting",
                        "Improve message personalization and clarity",
                        "Test different timing and frequency strategies",
                    ]

    def _define_success_metrics(self, campaign):
        """Define success metrics for the campaign using LLM"""
        try:
            # Create detailed prompt for LLM to generate success metrics
            prompt = f"""
            Define appropriate success metrics for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Budget: {campaign.get('budget', 'Not specified')}
            - Goals: {campaign.get('goals', [])}
            - Target Segments: {campaign.get('target_segments', [])}
            - Industry: {campaign.get('industry', 'general')}
            
            Generate specific success metrics in JSON format:
            {{
                "metrics": [
                    "metric 1 with target value",
                    "metric 2 with target value",
                    "metric 3 with target value",
                    "metric 4 with target value"
                ]
            }}
            
            Consider:
            - Channel-specific metrics (email: open rate, CTR; SMS: response rate; social: engagement)
            - Campaign type metrics (promotional: conversion, ROI; awareness: reach, impressions)
            - Industry benchmarks and realistic targets
            - Budget-appropriate expectations
            - Measurable and actionable KPIs
            """

            response = self.llm_service.generate(prompt)
            metrics_data = self._parse_json_response(response, {"metrics": []})
            return metrics_data.get("metrics", [])

        except Exception as e:
            print(f"⚠️ Error generating LLM success metrics: {e}")
            # Fallback to basic LLM-generated metrics
            try:
                fallback_prompt = f"""
                Generate 4 basic success metrics for a {campaign.get('channel', 'email')} 
                {campaign.get('type', 'general')} marketing campaign. Return as JSON array of strings.
                """
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(fallback_response, [])
            except:
                # Final fallback - minimal dynamic response
                channel = campaign.get("channel", "email")
                return [
                    f"{channel.title()} delivery rate > 90%",
                    f"{channel.title()} engagement rate > 15%",
                    "Conversion rate > 2%",
                    "Cost per acquisition < budget target",
                ]

    def _assess_campaign_risks(self, campaign):
        """Assess potential risks and mitigation strategies using LLM"""
        try:
            # Create detailed prompt for LLM to generate risk assessment
            prompt = f"""
            Analyze potential risks and mitigation strategies for this marketing campaign:
            
            Campaign Details:
            - Name: {campaign.get('name', 'Unnamed Campaign')}
            - Channel: {campaign.get('channel', 'email')}
            - Type: {campaign.get('type', 'general')}
            - Budget: {campaign.get('budget', 'Not specified')}
            - Message: {campaign.get('message', 'Not specified')}
            - Target Segments: {campaign.get('target_segments', [])}
            - Goals: {campaign.get('goals', [])}
            
            Generate risk assessment in JSON format:
            {{
                "identified_risks": [
                    "specific risk 1",
                    "specific risk 2",
                    "specific risk 3"
                ],
                "mitigation_strategies": [
                    "mitigation strategy 1",
                    "mitigation strategy 2",
                    "mitigation strategy 3"
                ]
            }}
            
            Consider:
            - Channel-specific risks (deliverability, compliance, platform changes)
            - Message content risks (spam filters, customer fatigue, brand perception)
            - Budget and timing risks (market conditions, competition)
            - Audience risks (targeting accuracy, segment responsiveness)
            - Technical risks (tracking, attribution, data quality)
            - Regulatory and compliance risks
            """

            response = self.llm_service.generate(prompt)
            risks_data = self._parse_json_response(
                response, {"identified_risks": [], "mitigation_strategies": []}
            )
            return risks_data

        except Exception as e:
            print(f"⚠️ Error generating LLM risk assessment: {e}")
            # Fallback to basic LLM-generated risks
            try:
                fallback_prompt = f"""
                Generate 3 risks and 3 mitigation strategies for a {campaign.get('channel', 'email')} 
                marketing campaign. Return as JSON with 'identified_risks' and 'mitigation_strategies' arrays.
                """
                fallback_response = self.llm_service.generate(fallback_prompt)
                return self._parse_json_response(
                    fallback_response,
                    {"identified_risks": [], "mitigation_strategies": []},
                )
            except:
                # Final fallback - minimal dynamic response
                channel = campaign.get("channel", "email")
                return {
                    "identified_risks": [
                        f"{channel.title()} deliverability challenges",
                        "Audience engagement decline",
                        "Budget optimization issues",
                    ],
                    "mitigation_strategies": [
                        f"Monitor {channel} performance metrics",
                        "Implement A/B testing protocols",
                        "Establish feedback monitoring system",
                    ],
                }

    def _generate_campaign_predictions(self, campaigns, segments, goals):
        """Generate detailed performance predictions for each campaign"""
        predictions = {}

        for i, campaign in enumerate(campaigns):
            campaign_id = f"campaign_{i+1}"
            campaign_name = getattr(campaign, "name", f"Campaign {i+1}")

            # Calculate base metrics based on campaign characteristics
            target_segments = getattr(campaign, "target_segments", [])
            channels = getattr(campaign, "channels", ["email"])
            budget = getattr(campaign, "budget_estimate", 10000)
            if isinstance(budget, str):
                try:
                    budget = float(budget.replace("$", "").replace(",", ""))
                except:
                    budget = 10000

            # Estimate audience size based on segments
            total_audience = 0
            for segment_name in target_segments:
                # Find matching segment data
                for segment in segments:
                    segment_name_attr = (
                        getattr(segment, "name", "")
                        if hasattr(segment, "name")
                        else (
                            segment.get("name", "") if isinstance(
                                segment, dict) else ""
                        )
                    )
                    if segment_name_attr == segment_name:
                        # Remove size calculation - focus on segment quality over quantity
                        # total_audience calculation removed
                        break
                else:
                    # Default audience size if segment not found
                    total_audience += 1000

            if total_audience == 0:
                total_audience = 5000  # Default audience

            # Calculate predictions based on channel effectiveness and budget
            channel_multiplier = 1.0
            if "email" in channels:
                channel_multiplier += 0.2
            if "social_media" in channels:
                channel_multiplier += 0.3
            if "display_ads" in channels:
                channel_multiplier += 0.1

            # Budget impact on reach
            # Cap at 3x for very high budgets
            budget_factor = min(budget / 10000, 3.0)

            # Base conversion rates by channel
            base_ctr = 0.025  # 2.5% base CTR
            base_conversion_rate = 0.03  # 3% base conversion rate

            # Adjust based on campaign characteristics
            campaign_type = getattr(campaign, "type", "general")
            if campaign_type == "promotional":
                base_ctr *= 1.4
                base_conversion_rate *= 1.6
            elif campaign_type == "awareness":
                base_ctr *= 1.1
                base_conversion_rate *= 0.8

            # Calculate final metrics
            expected_reach = int(
                total_audience * channel_multiplier * budget_factor * 0.7
            )
            expected_ctr = min(
                base_ctr * channel_multiplier, 0.15)  # Cap at 15%
            expected_clicks = int(expected_reach * expected_ctr)
            expected_conversion_rate = min(
                base_conversion_rate * channel_multiplier, 0.12
            )  # Cap at 12%
            expected_conversions = int(
                expected_clicks * expected_conversion_rate)

            # Calculate ROI (assuming average order value of $75)
            avg_order_value = 75
            revenue = expected_conversions * avg_order_value
            expected_roi = (revenue - budget) / budget if budget > 0 else 0

            # Customer engagement score (0-100)
            engagement_score = min(
                85, int(50 + (channel_multiplier - 1)
                        * 30 + (budget_factor - 1) * 10)
            )

            predictions[campaign_id] = {
                "campaign_name": campaign_name,
                "expected_metrics": {
                    "total_conversions": expected_conversions,
                    "conversion_rate": round(expected_conversion_rate * 100, 2),
                    "ctr": round(expected_ctr * 100, 2),
                    "expected_roi": round(expected_roi * 100, 1),
                    "customer_engagement_score": engagement_score,
                    "expected_reach": expected_reach,
                    "expected_clicks": expected_clicks,
                    "estimated_revenue": round(revenue, 2),
                },
                "explanation": {
                    "conversion_reasoning": f"Based on {len(channels)} channel(s) targeting {len(target_segments)} segment(s) with ${budget:,} budget",
                    "ctr_reasoning": f"CTR optimized for {', '.join(channels)} channels with {campaign_type} campaign type",
                    "roi_reasoning": f"ROI calculated from {expected_conversions} conversions at ${avg_order_value} average order value",
                    "engagement_reasoning": f"Engagement score based on channel mix effectiveness and budget allocation",
                },
                "confidence_level": min(
                    95, int(70 + len(channels) * 5 + (budget_factor - 1) * 10)
                ),
            }

        return predictions

    def _identify_top_performing_campaigns(self, campaign_predictions):
        """Identify and rank top-performing campaigns based on predicted metrics"""
        if not campaign_predictions:
            return []

        # Calculate composite score for each campaign
        scored_campaigns = []
        for campaign_id, prediction in campaign_predictions.items():
            metrics = prediction["expected_metrics"]

            # Weighted scoring (ROI: 40%, Conversions: 30%, Engagement: 20%, CTR: 10%)
            composite_score = (
                metrics["expected_roi"] * 0.4
                + (metrics["total_conversions"] / 100) *
                0.3  # Normalize conversions
                + metrics["customer_engagement_score"] * 0.2
                + metrics["ctr"] * 10 * 0.1  # Scale CTR to similar range
            )

            scored_campaigns.append(
                {
                    "campaign_id": campaign_id,
                    "campaign_name": prediction["campaign_name"],
                    "composite_score": round(composite_score, 2),
                    "key_strengths": self._identify_campaign_strengths(metrics),
                    "predicted_metrics": metrics,
                }
            )

        # Sort by composite score (descending)
        scored_campaigns.sort(key=lambda x: x["composite_score"], reverse=True)

        return scored_campaigns[:3]  # Return top 3 campaigns

    def _identify_campaign_strengths(self, metrics):
        """Identify key strengths of a campaign based on its metrics"""
        strengths = []

        if metrics["expected_roi"] > 50:
            strengths.append("High ROI potential")
        if metrics["conversion_rate"] > 4:
            strengths.append("Strong conversion rate")
        if metrics["ctr"] > 3:
            strengths.append("Excellent click-through rate")
        if metrics["customer_engagement_score"] > 75:
            strengths.append("High customer engagement")
        if metrics["total_conversions"] > 50:
            strengths.append("High conversion volume")

        return strengths if strengths else ["Solid baseline performance"]

    def _generate_improvement_suggestions(self, campaigns, campaign_predictions):
        """Generate specific improvement suggestions for campaigns using LLM"""
        suggestions = []

        for i, campaign in enumerate(campaigns):
            campaign_id = f"campaign_{i+1}"
            prediction = campaign_predictions.get(campaign_id, {})
            metrics = prediction.get("expected_metrics", {})

            try:
                # Create detailed prompt for LLM to generate improvement suggestions
                prompt = f"""
                Analyze this marketing campaign and generate specific improvement suggestions:
                
                Campaign Details:
                - Name: {campaign.get('name', f'Campaign {i+1}')}
                - Type: {campaign.get('type', 'general')}
                - Channel: {campaign.get('channel', 'email')}
                - Budget: {campaign.get('budget', 'Not specified')}
                - Goals: {campaign.get('goals', [])}
                - Target Segments: {campaign.get('target_segments', [])}
                
                Current Performance Predictions:
                - Expected ROI: {metrics.get('expected_roi', 'N/A')}
                - Conversion Rate: {metrics.get('conversion_rate', 'N/A')}%
                - Click-Through Rate: {metrics.get('ctr', 'N/A')}%
                - Engagement Score: {metrics.get('customer_engagement_score', 'N/A')}
                
                Generate improvement suggestions in JSON format:
                {{
                    "campaign_suggestions": {{
                        "campaign_name": "{campaign.get('name', f'Campaign {i+1}')}",
                        "improvements": [
                            {{
                                "area": "improvement area",
                                "suggestion": "specific actionable suggestion",
                                "impact": "expected quantified impact",
                                "priority": "high/medium/low",
                                "implementation_effort": "low/medium/high"
                            }}
                        ]
                    }}
                }}
                
                Focus on:
                - Performance optimization opportunities
                - Channel and targeting improvements
                - Content and messaging enhancements
                - Budget allocation optimization
                - Timing and frequency adjustments
                - Personalization opportunities
                """

                response = self.llm_service.generate(prompt)
                suggestion_data = self._parse_json_response(
                    response, {"campaign_suggestions": {"improvements": []}}
                )
                campaign_suggestions = suggestion_data.get(
                    "campaign_suggestions", {"improvements": []}
                )

                if campaign_suggestions.get("improvements"):
                    suggestions.append(campaign_suggestions)

            except Exception as e:
                print(
                    f"⚠️ Error generating LLM improvement suggestions for campaign {i+1}: {e}"
                )
                # Fallback to basic LLM-generated suggestions
                try:
                    fallback_prompt = f"""
                    Generate 2-3 basic improvement suggestions for a {campaign.get('type', 'general')} 
                    campaign with current ROI of {metrics.get('expected_roi', 'unknown')}. 
                    Return as JSON with area, suggestion, and impact fields.
                    """
                    fallback_response = self.llm_service.generate(
                        fallback_prompt)
                    fallback_data = self._parse_json_response(
                        fallback_response, [])
                    if fallback_data:
                        suggestions.append(
                            {
                                "campaign_name": campaign.get(
                                    "name", f"Campaign {i+1}"
                                ),
                                "improvements": fallback_data,
                            }
                        )
                except:
                    # Enhanced fallback with LLM-generated content
                    try:
                        enhanced_fallback_prompt = f"""
                        Generate a specific improvement suggestion for campaign '{campaign.get('name', f'Campaign {i+1}')}' 
                        using {campaign.get('channel', 'email')} channel. Return JSON with area, suggestion, and impact fields.
                        Make it actionable and specific to the campaign details.
                        """
                        enhanced_response = self.llm_service.generate(
                            enhanced_fallback_prompt
                        )
                        enhanced_data = self._parse_json_response(
                            enhanced_response,
                            [
                                {
                                    "area": "General Optimization",
                                    "suggestion": f'Analyze and optimize {campaign.get("channel", "email")} campaign performance',
                                    "impact": "Potential 10-20% improvement in key metrics",
                                }
                            ],
                        )
                        suggestions.append(
                            {
                                "campaign_name": campaign.get(
                                    "name", f"Campaign {i+1}"
                                ),
                                "improvements": (
                                    enhanced_data
                                    if isinstance(enhanced_data, list)
                                    else [enhanced_data]
                                ),
                            }
                        )
                    except:
                        # Final static fallback only if LLM completely fails
                        suggestions.append(
                            {
                                "campaign_name": campaign.get(
                                    "name", f"Campaign {i+1}"
                                ),
                                "improvements": [
                                    {
                                        "area": "General Optimization",
                                        "suggestion": f'Analyze and optimize {campaign.get("channel", "email")} campaign performance',
                                        "impact": "Potential 10-20% improvement in key metrics",
                                    }
                                ],
                            }
                        )

        return suggestions

    def _enhance_decision(
        self, thinking_result: Dict, campaigns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prompt = f"Enhance decisioning based on thinking result: {thinking_result} and campaigns: {campaigns}. Generate dynamic recommendations."
        enhanced_str = self.thinking_layer.llm_service.generate(prompt)
        try:
            return json.loads(enhanced_str)
        except:
            fallback_prompt = "Generate a fallback decision with dynamic recommendations and confidence score."
            fallback_str = self.thinking_layer.llm_service.generate(
                fallback_prompt)
            try:
                return json.loads(fallback_str)
            except:
                confidence_prompt = (
                    "Generate a dynamic confidence score between 0.0 and 1.0."
                )
                confidence_str = self.thinking_layer.llm_service.generate(
                    confidence_prompt
                )
                try:
                    confidence = float(
                        json.loads(confidence_str).get("confidence", 0.0)
                    )
                except:
                    confidence = 0.0
                return {
                    "recommendations": [],
                    "confidence": confidence,
                }  # Dynamic confidence

    def _generate_ai_recommendation(self, thinking_result, customer_data):
        """Generate AI-powered recommendation based on LangChain thinking result and customer data"""
        # Extract AI insights from thinking result
        ai_insights = thinking_result.get("ai_insights", {})

        # Analyze customer preferences for channel optimization with AI enhancement
        channel_preferences = {}
        total_value = 0

        for customer in customer_data:
            pref_channel = customer.behavior.get("preferred_channel", "email")
            avg_value = customer.behavior.get("avg_order_value", 100)

            if pref_channel not in channel_preferences:
                channel_preferences[pref_channel] = {
                    "count": 0, "total_value": 0}

            channel_preferences[pref_channel]["count"] += 1
            channel_preferences[pref_channel]["total_value"] += avg_value
            total_value += avg_value

        # Calculate AI-optimized budget allocation
        budget_allocation = {}
        base_budget = 50000  # Default budget

        # Use AI preferences if available
        ai_channel_prefs = ai_insights.get("preferences", {}).get(
            "channel_preferences", {}
        )
        if ai_channel_prefs:
            for channel, ai_weight in ai_channel_prefs.items():
                budget_allocation[channel] = int(base_budget * ai_weight)
        else:
            # Fallback to customer data analysis
            for channel, data in channel_preferences.items():
                weight = (
                    (data["total_value"] /
                     total_value) if total_value > 0 else 0.25
                )
                budget_allocation[channel] = int(base_budget * weight)

        # Generate customer segment insights
        segments = set()
        for customer in customer_data:
            segments.update(customer.segments)

        return {
            "campaign_strategy": f"AI-optimized multi-channel approach targeting {len(segments)} segments",
            "budget_allocation": budget_allocation,
            "predicted_outcomes": thinking_result,
            "ai_confidence": thinking_result.get("simulation_confidence", 0.8),
            "timeline": "4-6 weeks (AI-optimized)",
            "success_metrics": [
                f"ROI > {thinking_result.get('predicted_roi', 2.5)}",
                f"Reach > {thinking_result.get('expected_reach', len(customer_data) * 1000)}",
                f"Conversion > {thinking_result.get('conversion_rate', 0.05) * 100:.1f}%",
            ],
            "target_segments": list(segments),
            "ai_recommendations": thinking_result.get("recommendations", []),
            "customer_insights": {
                "total_customers": len(customer_data),
                "avg_order_value": (
                    total_value / len(customer_data) if customer_data else 0
                ),
                "preferred_channels": channel_preferences,
                "ai_enhanced": True,
            },
            "ai_insights": ai_insights,
        }

    def _simulate_human_approval(self, approval_id):
        """Simulate human approval process with AI insights"""
        # In real system, this would be handled by actual human input
        print(
            f"{self.name}: Requesting human approval for AI-generated recommendation {approval_id}..."
        )

        # Simulate approval with AI-informed modifications
        modifications = {
            "budget_adjustment": "AI suggests increasing social media budget by 10%",
            "timeline_change": "AI recommends extending to 5 weeks for optimal results",
            "ai_monitoring": "Enable continuous AI optimization during campaign",
        }

        self.human_loop.process_approval(approval_id, True, modifications)
        print(
            f"{self.name}: AI recommendation {approval_id} approved with AI-informed modifications"
        )

    def communicate_with_agent1(self, message):
        """Send message to Agent 1"""
        print(f"{self.name}: Sending message to Agent1: {message}")
        return f"Message sent to Agent1: {message}"

    def use_tools(self, tool_name, *args, **kwargs):
        """Access shared tools"""
        print(f"{self.name}: Using tool {tool_name} with args {args}")
        return f"Tool {tool_name} executed successfully"

    def continue_process(self):
        """Continue processing after tool usage"""
        print(f"{self.name}: Continuing campaign workflow process...")
        return self.process()

    def run_campaign_workflow(self):
        """Run the complete AI-powered campaign workflow"""
        print(f"\n=== {self.name} AI-Powered Campaign Management Workflow ===")
        print("AI-Enhanced Workflow Components:")
        print("1. Campaign Data Input (goals, budget, channels, messaging, timing)")
        print("2. Customer Data Input (segments, behavior, interactions)")
        print("3. AI-Enhanced Context Engineering & Knowledge Base")
        print("4. LangChain-Powered Agent Thinking Layer:")
        print("   - AI Analysis Node (LangChain + OpenAI)")
        print("   - AI Reasoning Node (Strategic AI reasoning)")
        print("   - AI Preference Node (AI preference analysis)")
        print("   - AI Simulation Node (AI outcome prediction)")
        print("5. AI-Generated Recommendation")
        print("6. Human-in-the-Loop Approval (with AI insights)")
        print("7. AI-Optimized Campaign Execution Plan")
        print("8. Continuous AI Learning & Knowledge Base Updates")
        print("\n🤖 Starting AI-powered workflow execution...")

        return self.process()

    def _get_contextual_information(
        self, campaign_data: Dict[str, Any], customer_data: Any, user_query: str
    ) -> Dict[str, Any]:
        """Get contextual information from memory and retrieval systems"""
        contextual_info = {
            "conversation_history": "",
            "relevant_knowledge": [],
            "campaign_insights": [],
            "customer_insights": [],
            "timestamp": datetime.now().isoformat(),
        }

        if (
            hasattr(self.thinking_layer, "memory_system")
            and self.thinking_layer.memory_system
        ):
            # Get conversation context
            contextual_info["conversation_history"] = (
                self.thinking_layer.memory_system.get_conversation_context()
            )

            # Get general contextual information
            if user_query:
                general_context = (
                    self.thinking_layer.memory_system.get_contextual_information(
                        user_query
                    )
                )
                contextual_info["relevant_knowledge"] = general_context.get(
                    "relevant_knowledge", []
                )

        if (
            hasattr(self.thinking_layer, "retrieval_system")
            and self.thinking_layer.retrieval_system
        ):
            # Get campaign-specific insights
            campaign_insights = (
                self.thinking_layer.retrieval_system.retrieve_campaign_insights(
                    campaign_data
                )
            )
            contextual_info["campaign_insights"] = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in campaign_insights
            ]

            # Get customer-specific insights
            customer_insights = (
                self.thinking_layer.retrieval_system.retrieve_customer_insights(
                    customer_data
                )
            )
            contextual_info["customer_insights"] = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in customer_insights
            ]

        return contextual_info

    def _prepare_ai_context(
        self, campaign_data, customer_data, contextual_info: Dict[str, Any] = None
    ):
        """Prepare AI-enhanced context for thinking layer processing with memory integration"""
        base_context = self.context_store.get_relevant_context(campaign_data)

        # Add AI-specific context enhancements
        ai_context = {
            **base_context,
            "timestamp": datetime.now().isoformat(),
            "market_conditions": {
                "competition_level": "moderate",
                "ai_adoption_rate": "high",
                "personalization_demand": "increasing",
            },
            "customer_insights": {
                "total_profiles": (
                    1 if not isinstance(
                        customer_data, list) else len(customer_data)
                ),
                "ai_personalization_readiness": "high",
                "behavioral_patterns": (
                    [customer_data.behavior]
                    if not isinstance(customer_data, list)
                    else [customer.behavior for customer in customer_data]
                ),
            },
            "ai_capabilities": {
                "langchain_integration": True,
                "openai_powered": True,
                "continuous_learning": True,
            },
        }

        # Add contextual information if available
        if contextual_info:
            ai_context["contextual_memory"] = {
                "conversation_history": contextual_info.get("conversation_history", ""),
                "relevant_knowledge_count": len(
                    contextual_info.get("relevant_knowledge", [])
                ),
                "campaign_insights_count": len(
                    contextual_info.get("campaign_insights", [])
                ),
                "customer_insights_count": len(
                    contextual_info.get("customer_insights", [])
                ),
            }

            # Add specific insights to context
            if contextual_info.get("campaign_insights"):
                ai_context["retrieved_campaign_insights"] = [
                    insight["content"]
                    for insight in contextual_info["campaign_insights"][:3]
                ]

            if contextual_info.get("customer_insights"):
                ai_context["retrieved_customer_insights"] = [
                    insight["content"]
                    for insight in contextual_info["customer_insights"][:3]
                ]

        return ai_context

    def _update_knowledge_base_with_ai_insights(self, thinking_result):
        """Update knowledge base with AI-generated insights"""
        ai_insights = thinking_result.get("ai_insights", {})

        # Store AI insights in knowledge base
        insight_key = f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.knowledge_base[insight_key] = {
            "timestamp": datetime.now().isoformat(),
            "ai_analysis": ai_insights.get("analysis", {}),
            "ai_reasoning": ai_insights.get("reasoning", {}),
            "ai_preferences": ai_insights.get("preferences", {}),
            "ai_simulation": ai_insights.get("simulation", {}),
            "confidence_score": thinking_result.get("simulation_confidence", 0.8),
            "recommendations": thinking_result.get("recommendations", []),
        }

        print(f"{self.name}: Updated knowledge base with AI insights ({insight_key})")

    def get_ai_knowledge_summary(self):
        """Get summary of AI-generated knowledge"""
        ai_entries = [
            k for k in self.knowledge_base.keys() if k.startswith("ai_insights_")
        ]

        return {
            "total_ai_insights": len(ai_entries),
            "knowledge_base_size": len(self.knowledge_base),
            "ai_learning_active": True,
            "langchain_integration": "active",
            "last_ai_update": max(
                [self.knowledge_base[k].get("timestamp", "")
                 for k in ai_entries],
                default="never",
            ),
        }

    def process_agent1_output(
        self, optimized_campaigns: List[Dict[str, Any]], customer_data: Dict
    ) -> Dict[str, Any]:
        """Process optimized output from Agent1, enhancing thinking and decisioning."""
        context = self._prepare_ai_context(
            optimized_campaigns, customer_data, {})
        thinking_result = self.thinking_layer.process(
            optimized_campaigns, customer_data, context
        )
        enhanced_decision = self._enhance_decision(
            thinking_result, optimized_campaigns)
        return enhanced_decision

    def _enhance_decision(
        self, thinking_result: Dict, campaigns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prompt = f"Enhance decisioning based on thinking result: {thinking_result} and campaigns: {campaigns}. Generate dynamic recommendations."
        enhanced_str = self.thinking_layer.llm_service.generate(prompt)
        try:
            return json.loads(enhanced_str)
        except:
            fallback_prompt = "Generate a fallback decision with dynamic recommendations and confidence score."
            fallback_str = self.thinking_layer.llm_service.generate(
                fallback_prompt)
            try:
                return json.loads(fallback_str)
            except:
                confidence_prompt = (
                    "Generate a dynamic confidence score between 0.0 and 1.0."
                )
                confidence_str = self.thinking_layer.llm_service.generate(
                    confidence_prompt
                )
                try:
                    confidence = float(
                        json.loads(confidence_str).get("confidence", 0.0)
                    )
                except:
                    confidence = 0.0
                return {
                    "recommendations": [],
                    "confidence": confidence,
                }  # Dynamic confidence

    def _generate_ai_recommendation(self, thinking_result, customer_data):
        """Generate AI-powered recommendation based on LangChain thinking result and customer data"""
        # Extract AI insights from thinking result
        ai_insights = thinking_result.get("ai_insights", {})

        # Analyze customer preferences for channel optimization with AI enhancement
        channel_preferences = {}
        total_value = 0

        for customer in customer_data:
            pref_channel = customer.behavior.get("preferred_channel", "email")
            avg_value = customer.behavior.get("avg_order_value", 100)

            if pref_channel not in channel_preferences:
                channel_preferences[pref_channel] = {
                    "count": 0, "total_value": 0}

            channel_preferences[pref_channel]["count"] += 1
            channel_preferences[pref_channel]["total_value"] += avg_value
            total_value += avg_value

        # Calculate AI-optimized budget allocation
        budget_allocation = {}
        base_budget = 50000  # Default budget

        # Use AI preferences if available
        ai_channel_prefs = ai_insights.get("preferences", {}).get(
            "channel_preferences", {}
        )
        if ai_channel_prefs:
            for channel, ai_weight in ai_channel_prefs.items():
                budget_allocation[channel] = int(base_budget * ai_weight)
        else:
            # Fallback to customer data analysis
            for channel, data in channel_preferences.items():
                weight = (
                    (data["total_value"] /
                     total_value) if total_value > 0 else 0.25
                )
                budget_allocation[channel] = int(base_budget * weight)

        # Generate customer segment insights
        segments = set()
        for customer in customer_data:
            segments.update(customer.segments)

        return {
            "campaign_strategy": f"AI-optimized multi-channel approach targeting {len(segments)} segments",
            "budget_allocation": budget_allocation,
            "predicted_outcomes": thinking_result,
            "ai_confidence": thinking_result.get("simulation_confidence", 0.8),
            "timeline": "4-6 weeks (AI-optimized)",
            "success_metrics": [
                f"ROI > {thinking_result.get('predicted_roi', 2.5)}",
                f"Reach > {thinking_result.get('expected_reach', len(customer_data) * 1000)}",
                f"Conversion > {thinking_result.get('conversion_rate', 0.05) * 100:.1f}%",
            ],
            "target_segments": list(segments),
            "ai_recommendations": thinking_result.get("recommendations", []),
            "customer_insights": {
                "total_customers": len(customer_data),
                "avg_order_value": (
                    total_value / len(customer_data) if customer_data else 0
                ),
                "preferred_channels": channel_preferences,
                "ai_enhanced": True,
            },
            "ai_insights": ai_insights,
        }
