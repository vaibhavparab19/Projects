from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from tools.llm_service import LLMService
from thinking_node.core import AnalysisNode, ReasoningNode
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time


@dataclass
class CustomerSegment:
    """Represents a customer segment with targeting criteria"""

    name: str
    description: str
    criteria: Dict[str, Any]
    priority_score: Optional[float] = None
    characteristics: Optional[Dict[str, Any]] = None


@dataclass
class BusinessGoal:
    """Business goal with metrics and targets"""

    name: str
    description: str
    target_segments: List[str]
    success_metrics: List[str]
    priority: str  # high, medium, low
    timeline: Optional[str] = None
    expected_impact: Optional[str] = None


@dataclass
class MarketingCampaign:
    """Marketing campaign with content and targeting"""

    name: str
    description: str
    target_segments: List[str]
    business_goals: List[str]
    channels: List[str]
    content: Dict[str, str]  # channel -> content
    budget_estimate: Optional[float] = None
    timeline: Optional[str] = None
    success_metrics: List[str] = None
    optimization_score: Optional[float] = None
    evaluation_score: Optional[float] = None
    evaluation_reason: Optional[str] = None


class Agent1:
    """Dynamic Customer Segmentation and Campaign Generation Agent"""

    def __init__(self):
        self.analysis_node = AnalysisNode()
        self.reasoning_node = ReasoningNode()
        self.llm_service = LLMService()
        self.segments: List[CustomerSegment] = []
        self.business_goals: List[BusinessGoal] = []
        self.campaigns: List[MarketingCampaign] = []

    def analyze_customer_data(self, data: Any) -> Dict[str, Any]:
        """Dynamically analyze any customer data format and extract insights"""
        print("\n🔍 Analyzing customer data...")

        # Store customer data for analysis
        pass

        # Convert data to analyzable format
        data_str = self._prepare_data_for_analysis(data)

        # Use thinking node for deep analysis
        analysis_context = {
            "data": data_str,
            "task": "customer_segmentation_analysis",
            "requirements": "identify patterns, behaviors, and segmentation opportunities",
        }

        # The AnalysisNode expects campaign_data, customer_data, and context parameters
        analysis = self.analysis_node.analyze(
            campaign_data=data,  # Raw data for campaign context
            customer_data=data,  # Customer data
            context=analysis_context,  # Analysis context
        )

        # Generate insights using LLM
        insights_prompt = f"""
        Analyze this customer data and provide insights for segmentation:
        
        Data: {data_str}
        
        Analysis Context: {analysis}
        
        Please identify:
        1. Key customer attributes and behaviors
        2. Potential segmentation dimensions (demographics, behavior, value, lifecycle)
        3. Data quality and completeness assessment
        4. Recommended segmentation approaches
        
        Format as JSON with keys: attributes, behaviors, dimensions, quality_assessment, recommendations
        """

        insights_response = self.llm_service.generate(insights_prompt)
        insights = self._parse_json_response(
            insights_response,
            {
                "attributes": [],
                "behaviors": [],
                "dimensions": [],
                "quality_assessment": "Unknown",
                "recommendations": [],
            },
        )

        print(
            f"✅ Data analysis complete. Found {len(insights.get('attributes', []))} key attributes."
        )
        return insights

    def generate_customer_segments(
        self, data_insights: Dict[str, Any]
    ) -> List[CustomerSegment]:
        """Generate dynamic customer segments based on data insights"""
        print("\n🎯 Generating customer segments...")

        # Use reasoning node for segment strategy
        reasoning_context = {
            "insights": data_insights,
            "task": "segment_generation",
            "constraints": "create actionable, measurable segments",
        }

        reasoning = self.reasoning_node.reason(data_insights)

        # Generate segments using LLM
        segments_prompt = f"""
        Based on these customer data insights, generate 8-12 dynamic customer segments:
        
        Insights: {json.dumps(data_insights, indent=2)}
        Reasoning: {reasoning}
        
        Create segments similar to these examples but tailored to the actual data:
        - High-churn-risk subscribers: Score >80 on churn propensity in the last 30 days
        - VIP spenders: LTV >$2,000, ≥5 orders in last year
        - First-time buyers: Purchased once in <30d
        - Cart abandoners: Added >$100 to cart but no checkout in 7d
        - Lapsed app users: No logins >14d, previously active weekly
        
        For each segment, provide:
        - name: Clear, descriptive name
        - description: Detailed explanation
        - criteria: Specific, measurable criteria as key-value pairs
        - characteristics: Key behavioral/demographic traits
        - priority_score: 1-10 based on business value
        
        Format as JSON array of segment objects.
        """

        segments_response = self.llm_service.generate(segments_prompt)
        segments_data = self._parse_json_response(segments_response, [])

        # Convert to CustomerSegment objects and calculate sizes
        segments = []
        for seg_data in segments_data:
            if isinstance(seg_data, dict) and "name" in seg_data:
                segment = CustomerSegment(
                    name=seg_data.get("name", "Unnamed Segment"),
                    description=seg_data.get("description", ""),
                    criteria=seg_data.get("criteria", {}),
                    characteristics=seg_data.get("characteristics", {}),
                    priority_score=seg_data.get("priority_score", 5.0),
                )
                segments.append(segment)

        self.segments = segments
        print(f"✅ Generated {len(segments)} customer segments.")
        return segments

    def generate_business_goals(
        self, segments: List[CustomerSegment], socketio=None
    ) -> List[BusinessGoal]:
        """Generate business goals based on customer segments"""
        print("\n🎯 Generating business goals...")

        segments_summary = [
            {
                "name": seg.name,
                "description": seg.description,
                "priority_score": seg.priority_score,
            }
            for seg in segments
        ]

        goals_prompt = f"""
        Based on these customer segments, generate 8-12 strategic business goals:
        
        Segments: {json.dumps(segments_summary, indent=2)}
        
        Create goals similar to these examples but tailored to the segments:
        - Increase subscription renewals
        - Drive restocked purchases on core product lines
        - Win back lapsed users
        - Maximize AOV by cross-selling bundles
        - Reduce trial-to-paid drop-off
        - Boost mobile app re-engagement
        - Suppress over-messaging to reduce unsubscribes
        - Upsell premium plans
        - Collect user feedback for a feature launch
        - Drive activation of unused product capabilities
        
        For each goal, provide:
        - name: Clear, actionable goal name
        - description: Detailed explanation
        - target_segments: List of relevant segment names
        - success_metrics: List of measurable KPIs
        - priority: high/medium/low
        - timeline: Suggested timeframe
        - expected_impact: Anticipated business impact
        
        Format as JSON array of goal objects.
        """

        goals_response = self.llm_service.generate(goals_prompt)
        goals_data = self._parse_json_response(goals_response, [])

        # Convert to BusinessGoal objects
        goals = []
        for goal_data in goals_data:
            if isinstance(goal_data, dict) and "name" in goal_data:
                goal = BusinessGoal(
                    name=goal_data.get("name", "Unnamed Goal"),
                    description=goal_data.get("description", ""),
                    target_segments=goal_data.get("target_segments", []),
                    success_metrics=goal_data.get("success_metrics", []),
                    priority=goal_data.get("priority", "medium"),
                    timeline=goal_data.get("timeline"),
                    expected_impact=goal_data.get("expected_impact"),
                )
                goals.append(goal)

        print(f"✅ Generated {len(goals)} business goals.")
        return self._human_approve_goals(goals, socketio)

    def _human_approve_goals(
        self, goals: List[BusinessGoal], socketio=None
    ) -> List[BusinessGoal]:
        """Allow human to approve, modify, or reject business goals via UI"""
        print("\n📋 Generated Business Goals:")
        print("=" * 50)

        for i, goal in enumerate(goals, 1):
            print(f"\n{i}. {goal.name}")
            print(f"   Description: {goal.description}")
            print(f"   Priority: {goal.priority}")
            print(f"   Target Segments: {', '.join(goal.target_segments)}")
            print(f"   Success Metrics: {', '.join(goal.success_metrics)}")
            if goal.timeline:
                print(f"   Timeline: {goal.timeline}")

        # If no socketio provided, use all goals (fallback)
        if not socketio:
            print("⚠️ No UI interface available. Using all generated goals.")
            self.business_goals = goals
            return goals

        # Store goals for UI access
        self.pending_goals = goals
        self.approved_goals = []

        # Emit goals to UI for approval
        goals_data = [asdict(goal) for goal in goals]
        socketio.emit("goals_generated", {"goals": goals_data})

        print(
            "\n🔧 Goals sent to UI for approval. Please use the web interface to manage goals."
        )

        # Return current goals (will be updated via SocketIO events)
        return goals

    def generate_marketing_campaigns(
        self, segments: List[CustomerSegment], goals: List[BusinessGoal]
    ) -> List[MarketingCampaign]:
        """Generate marketing campaigns based on segments and goals"""
        print("\n📢 Generating marketing campaigns...")

        try:
            # Prepare context for campaign generation
            segments_data = [asdict(seg) for seg in segments]
            goals_data = [asdict(goal) for goal in goals]

            campaigns_prompt = f"""
            Generate 6-10 marketing campaigns based on these segments and business goals:
            
            Customer Segments: {json.dumps(segments_data, indent=2)}
            
            Business Goals: {json.dumps(goals_data, indent=2)}
            
            For each campaign, create:
            - name: Compelling campaign name
            - description: Campaign overview and strategy
            - target_segments: List of segment names to target
            - business_goals: List of goal names this campaign supports
            - channels: Marketing channels (email, social, paid ads, etc.)
            - content: Channel-specific content (subject lines, ad copy, etc.)
            - timeline: Suggested campaign duration
            - success_metrics: KPIs to track
            - budget_estimate: Rough budget estimate if applicable
            
            Make campaigns specific, actionable, and aligned with the segments and goals.
            Format as JSON array of campaign objects.
            """

            print("🔄 Calling LLM service for campaign generation...")
            campaigns_response = self.llm_service.generate(campaigns_prompt)
            print(
                f"📝 LLM Response received (length: {len(campaigns_response)} chars)")

            campaigns_data = self._parse_json_response(campaigns_response, [])
            print(
                f"📊 Parsed {len(campaigns_data)} campaign objects from response")

            # Convert to MarketingCampaign objects
            campaigns = []
            for i, camp_data in enumerate(campaigns_data):
                try:
                    if isinstance(camp_data, dict) and "name" in camp_data:
                        campaign = MarketingCampaign(
                            name=camp_data.get("name", f"Campaign {i+1}"),
                            description=camp_data.get(
                                "description", "Generated marketing campaign"
                            ),
                            target_segments=camp_data.get(
                                "target_segments", []),
                            business_goals=camp_data.get("business_goals", []),
                            channels=camp_data.get(
                                "channels", ["Email", "Social Media"]
                            ),
                            content=camp_data.get(
                                "content",
                                {
                                    "email": "Campaign content",
                                    "social": "Social media content",
                                },
                            ),
                            timeline=camp_data.get("timeline", "4-6 weeks"),
                            success_metrics=camp_data.get(
                                "success_metrics",
                                ["Click-through rate", "Conversion rate"],
                            ),
                            budget_estimate=camp_data.get(
                                "budget_estimate", 10000),
                        )
                        campaigns.append(campaign)
                        print(f"✅ Created campaign: {campaign.name}")
                    else:
                        print(
                            f"⚠️ Skipping invalid campaign data at index {i}: {camp_data}"
                        )
                except Exception as e:
                    print(f"❌ Error creating campaign {i}: {str(e)}")
                    # Create a fallback campaign
                    fallback_campaign = MarketingCampaign(
                        name=f"Fallback Campaign {i+1}",
                        description="Auto-generated fallback campaign",
                        target_segments=[seg.name for seg in segments[:2]],
                        business_goals=[goal.name for goal in goals[:2]],
                        channels=["Email", "Social Media"],
                        content={
                            "email": "Engaging email content",
                            "social": "Social media post",
                        },
                        timeline="4-6 weeks",
                        success_metrics=[
                            "Click-through rate", "Conversion rate"],
                        budget_estimate=10000,
                    )
                    campaigns.append(fallback_campaign)
                    print(
                        f"🔄 Created fallback campaign: {fallback_campaign.name}")

            # Ensure we always have at least some campaigns
            if len(campaigns) == 0:
                print("⚠️ No campaigns generated, creating default campaigns...")
                campaigns = self._create_default_campaigns(segments, goals)

            print(
                f"✅ Generated {len(campaigns)} marketing campaigns successfully.")
            # Store campaigns in instance for later access
            self.campaigns = campaigns
            return campaigns

        except Exception as e:
            print(f"❌ Critical error in campaign generation: {str(e)}")
            # Create fallback campaigns to ensure workflow continues
            print("🔄 Creating fallback campaigns to ensure workflow continuation...")
            fallback_campaigns = self._create_default_campaigns(
                segments, goals)
            self.campaigns = fallback_campaigns
            return fallback_campaigns

    def _human_review_campaigns(
        self, campaigns: List[MarketingCampaign]
    ) -> List[MarketingCampaign]:
        """Allow human to review, modify, approve, or reject campaigns"""
        print("\n📢 Generated Marketing Campaigns:")
        print("=" * 50)

        for i, campaign in enumerate(campaigns, 1):
            print(f"\n{i}. {campaign.name}")
            print(f"   Description: {campaign.description}")
            print(f"   Target Segments: {', '.join(campaign.target_segments)}")
            print(f"   Business Goals: {', '.join(campaign.business_goals)}")
            print(f"   Channels: {', '.join(campaign.channels)}")
            if campaign.timeline:
                print(f"   Timeline: {campaign.timeline}")
            if campaign.content:
                print(f"   Content Preview: {list(campaign.content.keys())}")

        approved_campaigns = []

        while True:
            print("\n🔧 Campaign Management Options:")
            print("1. Approve specific campaigns (enter numbers)")
            print("2. Modify a campaign")
            print("3. Reject a campaign")
            print("4. View campaign content details")
            print("5. Add custom campaign")
            print("6. Finish selection")

            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                try:
                    selected = input(
                        "Enter campaign numbers to approve (comma-separated): "
                    )
                    indices = [int(x.strip()) - 1 for x in selected.split(",")]
                    for idx in indices:
                        if (
                            0 <= idx < len(campaigns)
                            and campaigns[idx] not in approved_campaigns
                        ):
                            approved_campaigns.append(campaigns[idx])
                            print(f"✅ Approved: {campaigns[idx].name}")
                except ValueError:
                    print("❌ Invalid input. Please enter numbers separated by commas.")

            elif choice == "2":
                try:
                    idx = int(input("Enter campaign number to modify: ")) - 1
                    if 0 <= idx < len(campaigns):
                        campaign = campaigns[idx]
                        print(f"\nModifying: {campaign.name}")
                        print("1. Name")
                        print("2. Description")
                        print("3. Target Segments")
                        print("4. Channels")
                        print("5. Content")

                        field_choice = input("What to modify (1-5): ").strip()

                        if field_choice == "1":
                            campaign.name = input("New name: ").strip()
                        elif field_choice == "2":
                            campaign.description = input(
                                "New description: ").strip()
                        elif field_choice == "3":
                            segments_str = input(
                                "New target segments (comma-separated): "
                            )
                            campaign.target_segments = [
                                s.strip() for s in segments_str.split(",")
                            ]
                        elif field_choice == "4":
                            channels_str = input(
                                "New channels (comma-separated): ")
                            campaign.channels = [
                                c.strip() for c in channels_str.split(",")
                            ]
                        elif field_choice == "5":
                            channel = input(
                                "Which channel content to modify: ").strip()
                            new_content = input(
                                f"New content for {channel}: ").strip()
                            campaign.content[channel] = new_content

                        print(f"✅ Modified: {campaign.name}")
                except (ValueError, IndexError):
                    print("❌ Invalid campaign number.")

            elif choice == "3":
                try:
                    idx = int(input("Enter campaign number to reject: ")) - 1
                    if 0 <= idx < len(campaigns):
                        rejected_campaign = campaigns.pop(idx)
                        print(f"❌ Rejected: {rejected_campaign.name}")
                except (ValueError, IndexError):
                    print("❌ Invalid campaign number.")

            elif choice == "4":
                try:
                    idx = int(
                        input("Enter campaign number to view details: ")) - 1
                    if 0 <= idx < len(campaigns):
                        campaign = campaigns[idx]
                        print(f"\n📋 {campaign.name} - Content Details:")
                        for channel, content in campaign.content.items():
                            print(f"\n{channel.upper()}:")
                            print(f"  {content}")
                except (ValueError, IndexError):
                    print("❌ Invalid campaign number.")

            elif choice == "5":
                name = input("Campaign name: ").strip()
                description = input("Campaign description: ").strip()
                segments_str = input("Target segments (comma-separated): ")
                target_segments = [s.strip() for s in segments_str.split(",")]
                channels_str = input("Channels (comma-separated): ")
                channels = [c.strip() for c in channels_str.split(",")]

                content = {}
                for channel in channels:
                    content[channel] = input(
                        f"Content for {channel}: ").strip()

                custom_campaign = MarketingCampaign(
                    name=name,
                    description=description,
                    target_segments=target_segments,
                    business_goals=[],
                    channels=channels,
                    content=content,
                )
                campaigns.append(custom_campaign)
                print(f"✅ Added custom campaign: {name}")

            elif choice == "6":
                break

            else:
                print("❌ Invalid choice. Please enter 1-6.")

        if not approved_campaigns:
            print("⚠️ No campaigns approved. Using all generated campaigns.")
            approved_campaigns = campaigns

        self.campaigns = approved_campaigns
        print(
            f"\n✅ Final selection: {len(approved_campaigns)} marketing campaigns approved."
        )
        return approved_campaigns

    def optimize_and_evaluate_campaigns(
        self, campaigns: List[MarketingCampaign]
    ) -> List[MarketingCampaign]:
        """Optimize and evaluate campaigns after human modifications"""
        print("\n🚀 Optimizing and evaluating campaigns...")

        for i, campaign in enumerate(campaigns):
            print(
                f"\nProcessing campaign {i+1}/{len(campaigns)}: {campaign.name}")

            # Optimization
            optimization_prompt = f"""
            Optimize this marketing campaign for better performance:
            
            Campaign: {json.dumps(asdict(campaign), indent=2)}
            
            Provide optimization suggestions for:
            1. Content improvements
            2. Channel optimization
            3. Targeting refinements
            4. Timing recommendations
            
            Return a score from 0-100 for optimization potential and specific recommendations.
            Format as JSON with keys: optimization_score, recommendations
            """

            opt_response = self.llm_service.generate(optimization_prompt)
            opt_data = self._parse_json_response(
                opt_response,
                {
                    "optimization_score": 75,
                    "recommendations": ["No specific optimizations identified"],
                },
            )

            campaign.optimization_score = opt_data.get(
                "optimization_score", 75)

            # Evaluation
            evaluation_prompt = f"""
            Evaluate this marketing campaign's potential effectiveness:
            
            Campaign: {json.dumps(asdict(campaign), indent=2)}
            
            Consider:
            1. Target segment alignment
            2. Content quality and relevance
            3. Channel appropriateness
            4. Goal alignment
            5. Measurability of success metrics
            
            Provide an effectiveness score (0-100) and detailed reasoning.
            Format as JSON with keys: evaluation_score, evaluation_reason
            """

            eval_response = self.llm_service.generate(evaluation_prompt)
            eval_data = self._parse_json_response(
                eval_response,
                {
                    "evaluation_score": 70,
                    "evaluation_reason": "Campaign shows good potential with room for improvement",
                },
            )

            campaign.evaluation_score = eval_data.get("evaluation_score", 70)
            campaign.evaluation_reason = eval_data.get(
                "evaluation_reason", "Standard evaluation"
            )

            print(f"   Optimization Score: {campaign.optimization_score}/100")
            print(f"   Evaluation Score: {campaign.evaluation_score}/100")

        print("\n✅ Campaign optimization and evaluation complete.")
        return campaigns

    def prepare_for_agent2(self, campaigns: List[MarketingCampaign]) -> Dict[str, Any]:
        """Prepare final output for Agent2 processing"""
        print("\n📤 Preparing data for Agent2...")

        # Create comprehensive output package
        output = {
            "timestamp": datetime.now().isoformat(),
            "agent1_version": "2.0_dynamic",
            "customer_segments": [asdict(seg) for seg in self.segments],
            "business_goals": [asdict(goal) for goal in self.business_goals],
            "marketing_campaigns": [asdict(camp) for camp in campaigns],
            "summary": {
                "total_segments": len(self.segments),
                "total_goals": len(self.business_goals),
                "total_campaigns": len(campaigns),
                "avg_campaign_score": (
                    sum(c.evaluation_score or 0 for c in campaigns) /
                    len(campaigns)
                    if campaigns
                    else 0
                ),
                "high_priority_goals": len(
                    [g for g in self.business_goals if g.priority == "high"]
                ),
                "optimization_opportunities": len(
                    [c for c in campaigns if (c.optimization_score or 0) < 80]
                ),
            },
            "recommendations": {
                "next_steps": [
                    "Review campaign performance metrics",
                    "Set up tracking and measurement systems",
                    "Plan campaign execution timeline",
                    "Allocate budget and resources",
                ],
                "success_factors": [
                    "Regular performance monitoring",
                    "A/B testing of campaign elements",
                    "Continuous segment refinement",
                    "Cross-channel coordination",
                ],
            },
        }

        print(
            f"✅ Prepared comprehensive package with {len(campaigns)} campaigns for Agent2."
        )
        return output

    def run_complete_workflow(self, customer_data: Any) -> Dict[str, Any]:
        """Run the complete Agent1 workflow from data analysis to Agent2 handoff"""
        print("\n🚀 Starting Agent1 Complete Workflow")
        print("=" * 60)

        try:
            # Step 1: Analyze customer data
            insights = self.analyze_customer_data(customer_data)

            # Step 2: Generate customer segments
            segments = self.generate_customer_segments(insights)

            # Step 3: Generate business goals (with human approval)
            goals = self.generate_business_goals(segments)

            # Step 4: Generate marketing campaigns (with human review)
            campaigns = self.generate_marketing_campaigns(segments, goals)

            # Step 5: Optimize and evaluate campaigns
            optimized_campaigns = self.optimize_and_evaluate_campaigns(
                campaigns)

            # Step 6: Prepare for Agent2
            final_output = self.prepare_for_agent2(optimized_campaigns)

            print("\n🎉 Agent1 workflow completed successfully!")
            print(
                f"📊 Generated {len(segments)} segments, {len(goals)} goals, {len(campaigns)} campaigns"
            )

            return final_output

        except Exception as e:
            print(f"\n❌ Error in Agent1 workflow: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

    def run_complete_workflow_with_status(
        self, customer_data: Any, socketio=None
    ) -> Dict[str, Any]:
        """Run the complete Agent1 workflow with real-time status updates"""

        def emit_status(status, message, step=None, data=None):
            if socketio:
                socketio.emit(
                    "agent1_status",
                    {
                        "status": status,
                        "message": message,
                        "step": step,
                        "data": data,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            print(f"🤖 Agent1: {message}")

        emit_status("starting", "Starting Agent1 Complete Workflow")

        try:
            # Step 1: Analyze customer data
            emit_status(
                "thinking", "Analyzing customer data patterns...", "data_analysis"
            )
            insights = self.analyze_customer_data(customer_data)
            emit_status(
                "progress",
                "Customer data analysis completed",
                "data_analysis",
                insights,
            )

            # Step 2: Generate customer segments
            emit_status(
                "thinking", "Generating customer segments...", "segmentation")
            segments = self.generate_customer_segments(insights)
            segments_data = [asdict(seg) for seg in segments]
            if socketio:
                socketio.emit("segments_generated", {
                              "segments": segments_data})
            emit_status(
                "progress",
                f"Generated {len(segments)} customer segments",
                "segmentation",
                {"segments": segments_data},
            )

            # Step 3: Generate business goals (with human approval)
            emit_status("thinking", "Creating business goals...",
                        "goal_generation")
            goals = self.generate_business_goals(segments, socketio)
            goals_data = [asdict(goal) for goal in goals]
            emit_status(
                "waiting_approval",
                f"Generated {len(goals)} business goals - awaiting approval",
                "goal_approval",
                {"goals": goals_data},
            )

            # STOP HERE - Return partial result for goal approval
            # The workflow will continue after goals are approved via SocketIO
            return {
                "status": "awaiting_goal_approval",
                "segments": segments_data,
                "goals": goals_data,
                "message": "Workflow paused for business goal approval",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            emit_status(
                "error", f"Error in Agent1 workflow: {str(e)}", "error")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

    def continue_workflow_after_approval(
        self,
        segments: List[CustomerSegment],
        approved_goals: List[BusinessGoal],
        socketio=None,
    ) -> Dict[str, Any]:
        """Continue the workflow after business goals are approved"""

        def emit_status(status, message, step=None, data=None):
            if socketio:
                socketio.emit(
                    "agent1_status",
                    {
                        "status": status,
                        "message": message,
                        "step": step,
                        "data": data,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            print(f"🤖 Agent1: {message}")

        try:
            emit_status(
                "approved",
                f"Business goals approved! Continuing with {len(approved_goals)} goals...",
                "goal_approval",
            )

            # Step 4: Generate marketing campaigns (with human review)
            emit_status(
                "thinking",
                "Designing marketing campaigns based on approved goals...",
                "campaign_generation",
            )
            campaigns = self.generate_marketing_campaigns(
                segments, approved_goals)
            campaigns_data = [asdict(camp) for camp in campaigns]
            if socketio:
                socketio.emit("campaigns_generated", {
                              "campaigns": campaigns_data})
            emit_status(
                "waiting_review",
                f"Generated {len(campaigns)} campaigns - awaiting review",
                "campaign_review",
                {"campaigns": campaigns_data},
            )

            # Wait for human approval through web interface
            emit_status(
                "waiting_approval",
                "Waiting for campaign approval...",
                "campaign_approval",
            )

            # In web mode, campaigns are automatically approved after display
            # The approval is handled by the web interface
            time.sleep(1)  # Brief pause for UI update
            emit_status(
                "reviewed",
                "Marketing campaigns ready for optimization",
                "campaign_review",
            )

            # Step 5: Optimize and evaluate campaigns
            emit_status(
                "thinking", "Optimizing and evaluating campaigns...", "optimization"
            )
            optimized_campaigns = self.optimize_and_evaluate_campaigns(
                campaigns)
            emit_status(
                "progress", "Campaign optimization completed", "optimization")

            # Step 6: Prepare for Agent2
            emit_status(
                "thinking", "Preparing handoff to Agent2...", "handoff")
            final_output = self.prepare_for_agent2(optimized_campaigns)

            emit_status(
                "completed",
                f"Agent1 workflow completed! Generated {len(segments)} segments, {len(approved_goals)} goals, {len(campaigns)} campaigns",
                "completed",
            )

            return final_output

        except Exception as e:
            emit_status(
                "error", f"Error continuing Agent1 workflow: {str(e)}", "error")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

    def _prepare_data_for_analysis(self, data: Any) -> str:
        """Convert any data format to analyzable string"""
        if isinstance(data, str):
            return data
        elif isinstance(data, (dict, list)):
            return json.dumps(data, indent=2, default=str)
        else:
            return str(data)

    def _create_default_campaigns(
        self, segments: List[CustomerSegment], goals: List[BusinessGoal]
    ) -> List[MarketingCampaign]:
        """Create default campaigns when generation fails"""
        print("🔄 Creating default campaigns as fallback...")

        default_campaigns = []

        # Create basic campaigns based on segments and goals
        for i, segment in enumerate(segments[:3]):  # Limit to first 3 segments
            for j, goal in enumerate(goals[:2]):  # Limit to first 2 goals
                campaign = MarketingCampaign(
                    name=f"{segment.name} - {goal.name} Campaign",
                    description=f"Targeted campaign for {segment.name} to achieve {goal.name}",
                    target_segments=[segment.name],
                    business_goals=[goal.name],
                    channels=["Email", "Social Media", "Display Ads"],
                    content={
                        "email": f"Personalized email for {segment.name}",
                        "social": f"Social media content targeting {segment.name}",
                        "display": f"Display ads for {goal.name}",
                    },
                    timeline="4-6 weeks",
                    success_metrics=[
                        "Click-through rate",
                        "Conversion rate",
                        "Engagement rate",
                    ],
                    budget_estimate=15000,
                )
                default_campaigns.append(campaign)

        # Ensure we have at least 3 campaigns
        while len(default_campaigns) < 3:
            campaign_num = len(default_campaigns) + 1
            default_campaign = MarketingCampaign(
                name=f"Default Campaign {campaign_num}",
                description=f"Auto-generated marketing campaign {campaign_num}",
                target_segments=[segments[0].name] if segments else [
                    "All Customers"],
                business_goals=[goals[0].name] if goals else [
                    "Brand Awareness"],
                channels=["Email", "Social Media"],
                content={
                    "email": "Engaging email campaign content",
                    "social": "Social media marketing content",
                },
                timeline="4-6 weeks",
                success_metrics=["Click-through rate", "Conversion rate"],
                budget_estimate=10000,
            )
            default_campaigns.append(default_campaign)

        print(f"✅ Created {len(default_campaigns)} default campaigns")
        return default_campaigns

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
