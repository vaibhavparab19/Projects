"""LangGraph Multi-Agent Workflow Implementation

Implements a LangGraph-based workflow for coordinating Agent1 and Agent2
with the thinking layer and human-in-the-loop approval process.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
import json

# Import our agents and components
from agents.agent1 import Agent1
from agents.agent2 import Agent2
from agents.reflect_agent import ReflectAgent
from thinking_node.core import ThinkingLayer
from tools.llm_service import LLMService
from config import Config


class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow"""

    messages: Annotated[List[BaseMessage], "The conversation messages"]
    campaign_data: Dict[str, Any]
    customer_data: List[Dict[str, Any]]
    agent1_output: Optional[Dict[str, Any]]
    agent2_output: Optional[Dict[str, Any]]
    reflect_agent_output: Optional[Dict[str, Any]]
    thinking_result: Optional[Dict[str, Any]]
    human_approval: Optional[Dict[str, Any]]
    final_recommendation: Optional[Dict[str, Any]]
    workflow_status: str
    confidence_score: float
    iteration_count: int


class MultiAgentWorkflow:
    """LangGraph-powered multi-agent workflow coordinator"""

    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            api_key=self.config.OPENAI_API_KEY,
            temperature=0.7,
        )

        # Initialize agents
        self.agent1 = Agent1()
        self.agent2 = Agent2()
        self.reflect_agent = ReflectAgent()
        self.thinking_layer = ThinkingLayer()
        self.llm_service = LLMService()

        # Build the workflow graph
        self.workflow = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges"""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("agent1_process", self._agent1_node)
        workflow.add_node("agent2_process", self._agent2_node)
        workflow.add_node("reflect_agent", self._reflect_agent_node)
        workflow.add_node("thinking_layer", self._thinking_layer_node)
        workflow.add_node("coordination", self._coordination_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("initialize")

        # Add edges - Modified flow: initialize → thinking_layer → agent1 → agent2 → reflect_agent → coordination
        workflow.add_edge("initialize", "thinking_layer")
        workflow.add_edge("thinking_layer", "agent1_process")
        workflow.add_edge("agent1_process", "agent2_process")
        workflow.add_edge("agent2_process", "reflect_agent")
        workflow.add_edge("reflect_agent", "coordination")

        # Conditional edge for human approval
        workflow.add_conditional_edges(
            "coordination",
            self._should_request_human_approval,
            {
                "approve": "human_approval",
                "finalize": "finalize",
                "error": "error_handler",
            },
        )

        workflow.add_edge("human_approval", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    def _initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow with campaign and customer data"""
        print("🚀 LangGraph Multi-Agent Workflow: Initializing...")

        # Add initialization message
        init_message = HumanMessage(
            content=f"Starting multi-agent campaign workflow at {datetime.now().isoformat()}"
        )

        state["messages"].append(init_message)
        state["workflow_status"] = "initialized"
        state["iteration_count"] = 0
        state["confidence_score"] = 0.0

        print(
            f"✅ Workflow initialized with {len(state['customer_data'])} customer profiles"
        )
        return state

    def _agent1_node(self, state: WorkflowState) -> WorkflowState:
        """Agent 1 processing node"""
        print("🤖 Agent 1: Processing campaign data...")

        try:
            # Process through Agent 1
            agent1_result = self.agent1.process_campaign_data(
                state["campaign_data"], state["customer_data"]
            )

            state["agent1_output"] = agent1_result

            # Add Agent 1 message
            agent1_message = AIMessage(
                content=f"Agent 1 completed processing. Generated {len(agent1_result.get('recommendations', []))} initial recommendations."
            )
            state["messages"].append(agent1_message)

            print(
                f"✅ Agent 1: Completed with {len(agent1_result.get('recommendations', []))} recommendations"
            )

        except Exception as e:
            print(f"❌ Agent 1: Error occurred - {str(e)}")
            state["workflow_status"] = "error"
            state["agent1_output"] = {"error": str(e)}

        return state

    def _agent2_node(self, state: WorkflowState) -> WorkflowState:
        """Agent 2 processing node with LangChain integration"""
        print("🤖 Agent 2: Processing with AI-powered thinking layer...")

        try:
            # Process through Agent 2 with LangChain integration
            agent2_result = self.agent2.process_campaign_request(
                state["campaign_data"],
                state["customer_data"][0] if state["customer_data"] else {},
            )

            state["agent2_output"] = agent2_result

            # Add Agent 2 message
            agent2_message = AIMessage(
                content=f"Agent 2 completed AI-powered analysis. Confidence: {agent2_result.get('ai_confidence', 0.8):.2f}"
            )
            state["messages"].append(agent2_message)

            print(
                f"✅ Agent 2: Completed with AI confidence {agent2_result.get('ai_confidence', 0.8):.2f}"
            )

        except Exception as e:
            print(f"❌ Agent 2: Error occurred - {str(e)}")
            state["workflow_status"] = "error"
            state["agent2_output"] = {"error": str(e)}

        return state

    def _thinking_layer_node(self, state: WorkflowState) -> WorkflowState:
        """LangChain-powered thinking layer processing node - Initial analysis before agents"""
        print("🧠 Thinking Layer: Providing initial AI analysis and guidance...")

        try:
            # Prepare initial context for thinking layer (no agent outputs yet)
            context = {
                "timestamp": datetime.now().isoformat(),
                "workflow_iteration": state["iteration_count"],
                "phase": "initial_analysis",
            }

            # Process through thinking layer for initial analysis
            thinking_result = self.thinking_layer.process(
                state["campaign_data"],
                state["customer_data"][0] if state["customer_data"] else {},
                context,
            )

            # Store initial thinking layer guidance for agents
            state["initial_thinking_guidance"] = {
                "analysis": thinking_result.analysis,
                "reasoning": thinking_result.reasoning,
                "preferences": thinking_result.preferences,
                "simulation": thinking_result.simulation,
                "confidence_score": thinking_result.confidence_score,
                "recommendations": thinking_result.recommendations,
                "guidance_for_agents": {
                    "agent1_focus": "Use this analysis to guide customer segmentation and goal generation",
                    "agent2_focus": "Leverage these insights for campaign strategy and context engineering",
                },
            }

            state["confidence_score"] = thinking_result.confidence_score

            # Add thinking layer message
            thinking_message = AIMessage(
                content=f"Thinking layer provided initial guidance. Confidence: {thinking_result.confidence_score:.2f}, Recommendations: {len(thinking_result.recommendations)}"
            )
            state["messages"].append(thinking_message)

            print(
                f"✅ Thinking Layer: Provided initial guidance with confidence {thinking_result.confidence_score:.2f}"
            )

        except Exception as e:
            print(f"❌ Thinking Layer: Error occurred - {str(e)}")
            state["workflow_status"] = "error"
            state["initial_thinking_guidance"] = {"error": str(e)}

        return state

    def _reflect_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Reflect Agent processing node for conversational interface and self-reflection"""
        print("🤔 Reflect Agent: Processing with self-reflection capabilities...")

        try:
            # Prepare context for reflect agent
            context = {
                "campaign_data": state["campaign_data"],
                "customer_data": state["customer_data"],
                "agent1_output": state["agent1_output"],
                "agent2_output": state["agent2_output"],
                "thinking_guidance": state.get("initial_thinking_guidance", {}),
                "workflow_status": state["workflow_status"],
                "iteration_count": state["iteration_count"],
            }

            # Process through Reflect Agent for conversational analysis
            reflect_result = self.reflect_agent.process_workflow_context(
                context,
                "Analyze the current workflow state and provide conversational insights",
            )

            state["reflect_agent_output"] = {
                "analysis": reflect_result.get("analysis", ""),
                "reflection": reflect_result.get("reflection", ""),
                "suggestions": reflect_result.get("suggestions", []),
                "confidence": reflect_result.get("confidence", 0.8),
                "conversational_summary": reflect_result.get(
                    "conversational_summary", ""
                ),
            }

            # Add Reflect Agent message
            reflect_message = AIMessage(
                content=f"Reflect Agent completed analysis with {len(reflect_result.get('suggestions', []))} suggestions. Confidence: {reflect_result.get('confidence', 0.8):.2f}"
            )
            state["messages"].append(reflect_message)

            print(
                f"✅ Reflect Agent: Completed with confidence {reflect_result.get('confidence', 0.8):.2f}"
            )

        except Exception as e:
            print(f"❌ Reflect Agent: Error occurred - {str(e)}")
            state["workflow_status"] = "error"
            state["reflect_agent_output"] = {"error": str(e)}

        return state

    def _coordination_node(self, state: WorkflowState) -> WorkflowState:
        """Coordination node that synthesizes all agent outputs"""
        print("🔄 Coordination: Synthesizing multi-agent outputs...")

        try:
            # Use LLM to coordinate and synthesize outputs
            coordination_prompt = f"""
            You are a coordination AI that synthesizes outputs from multiple agents and AI systems.
            
            Initial Thinking Layer Guidance: {json.dumps(state['initial_thinking_guidance'], indent=2)}
            Agent 1 Output: {json.dumps(state['agent1_output'], indent=2)}
            Agent 2 Output: {json.dumps(state['agent2_output'], indent=2)}
            Reflect Agent Output: {json.dumps(state['reflect_agent_output'], indent=2)}
            
            Synthesize these outputs into a coherent, actionable campaign strategy.
            Focus on:
            1. Combining the initial AI guidance with agent-specific outputs
            2. Incorporating Reflect Agent's conversational insights and self-reflection
            3. Resolving any conflicts between recommendations
            4. Providing a unified strategic direction
            5. Assessing overall confidence and risk
            
            Provide a structured synthesis that can guide final decision making.
            """

            coordination_response = self.llm_service.generate_response(
                coordination_prompt,
                "You are an expert campaign coordination AI that synthesizes multi-agent insights.",
            )

            # Create coordinated recommendation
            coordinated_recommendation = {
                "synthesis": coordination_response,
                "combined_confidence": self._calculate_combined_confidence(state),
                "unified_strategy": self._create_unified_strategy(state),
                "risk_assessment": self._assess_combined_risks(state),
                "next_steps": self._generate_next_steps(state),
                "coordination_timestamp": datetime.now().isoformat(),
            }

            state["final_recommendation"] = coordinated_recommendation
            state["workflow_status"] = "coordinated"

            # Add coordination message
            coord_message = AIMessage(
                content=f"Coordination completed. Combined confidence: {coordinated_recommendation['combined_confidence']:.2f}"
            )
            state["messages"].append(coord_message)

            print(
                f"✅ Coordination: Completed with combined confidence {coordinated_recommendation['combined_confidence']:.2f}"
            )

        except Exception as e:
            print(f"❌ Coordination: Error occurred - {str(e)}")
            state["workflow_status"] = "error"
            state["final_recommendation"] = {"error": str(e)}

        return state

    def _should_request_human_approval(self, state: WorkflowState) -> str:
        """Determine if human approval is needed"""
        if state["workflow_status"] == "error":
            return "error"

        # Request human approval if confidence is below threshold
        combined_confidence = state["final_recommendation"].get(
            "combined_confidence", 0.0
        )

        if combined_confidence < 0.75:
            return "approve"
        else:
            return "finalize"

    def _human_approval_node(self, state: WorkflowState) -> WorkflowState:
        """Human-in-the-loop approval node"""
        print("👤 Human Approval: Requesting human review...")

        # Simulate human approval process
        approval_result = self._simulate_human_approval(
            state["final_recommendation"])

        state["human_approval"] = approval_result

        # Update final recommendation based on human feedback
        if approval_result["approved"]:
            state["final_recommendation"].update(
                approval_result.get("modifications", {})
            )
            state["workflow_status"] = "approved"
        else:
            state["workflow_status"] = "rejected"

        # Add human approval message
        approval_message = HumanMessage(
            content=f"Human approval: {'Approved' if approval_result['approved'] else 'Rejected'}"
        )
        state["messages"].append(approval_message)

        print(
            f"✅ Human Approval: {'Approved' if approval_result['approved'] else 'Rejected'}"
        )
        return state

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow and prepare execution plan"""
        print("🎯 Finalization: Preparing final execution plan...")

        # Create final execution plan
        execution_plan = {
            "campaign_strategy": state["final_recommendation"],
            "execution_timeline": self._create_execution_timeline(),
            "monitoring_plan": self._create_monitoring_plan(),
            "success_metrics": self._define_success_metrics(state),
            "workflow_summary": {
                "total_messages": len(state["messages"]),
                "final_confidence": state["confidence_score"],
                "agents_involved": ["Agent1", "Agent2", "ThinkingLayer"],
                "human_approval_required": state["human_approval"] is not None,
                "completion_time": datetime.now().isoformat(),
            },
        }

        state["final_recommendation"]["execution_plan"] = execution_plan
        state["workflow_status"] = "completed"

        # Add finalization message
        final_message = AIMessage(
            content=f"Workflow completed successfully. Final confidence: {state['confidence_score']:.2f}"
        )
        state["messages"].append(final_message)

        print(
            f"✅ Workflow completed successfully with confidence {state['confidence_score']:.2f}"
        )
        return state

    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors"""
        print("❌ Error Handler: Processing workflow error...")

        error_summary = {
            "error_occurred": True,
            "error_stage": state["workflow_status"],
            "error_details": {
                "agent1_error": (
                    state["agent1_output"].get("error")
                    if state["agent1_output"]
                    else None
                ),
                "agent2_error": (
                    state["agent2_output"].get("error")
                    if state["agent2_output"]
                    else None
                ),
                "reflect_agent_error": (
                    state["reflect_agent_output"].get("error")
                    if state["reflect_agent_output"]
                    else None
                ),
                "thinking_error": (
                    state["thinking_result"].get("error")
                    if state["thinking_result"]
                    else None
                ),
            },
            "recovery_suggestions": [
                "Check API keys and configuration",
                "Verify input data format",
                "Review agent implementations",
                "Check LangChain dependencies",
            ],
        }

        state["final_recommendation"] = error_summary
        state["workflow_status"] = "failed"

        return state

    def _calculate_combined_confidence(self, state: WorkflowState) -> float:
        """Calculate combined confidence from all agents"""
        confidences = []

        if state["agent1_output"] and "confidence" in state["agent1_output"]:
            confidences.append(state["agent1_output"]["confidence"])

        if state["agent2_output"] and "ai_confidence" in state["agent2_output"]:
            confidences.append(state["agent2_output"]["ai_confidence"])

        if (
            state["reflect_agent_output"]
            and "confidence" in state["reflect_agent_output"]
        ):
            confidences.append(state["reflect_agent_output"]["confidence"])

        if state["thinking_result"] and "confidence_score" in state["thinking_result"]:
            confidences.append(state["thinking_result"]["confidence_score"])

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _create_unified_strategy(self, state: WorkflowState) -> Dict[str, Any]:
        """Create unified strategy from all agent outputs"""
        return {
            "approach": "Multi-agent AI-powered campaign optimization",
            "channels": self._merge_channel_recommendations(state),
            "budget_allocation": self._merge_budget_recommendations(state),
            "timeline": "4-6 weeks with AI monitoring",
            "personalization": "High (AI-driven)",
        }

    def _merge_channel_recommendations(self, state: WorkflowState) -> List[str]:
        """Merge channel recommendations from all sources"""
        channels = set()

        # From Agent 1
        if state["agent1_output"] and "recommended_channels" in state["agent1_output"]:
            channels.update(state["agent1_output"]["recommended_channels"])

        # From Agent 2
        if state["agent2_output"] and "recommended_channels" in state["agent2_output"]:
            channels.update(state["agent2_output"]["recommended_channels"])

        # From Thinking Layer
        if state["thinking_result"] and "preferences" in state["thinking_result"]:
            prefs = state["thinking_result"]["preferences"].get(
                "channel_preferences", {}
            )
            channels.update(prefs.keys())

        return list(channels)

    def _merge_budget_recommendations(self, state: WorkflowState) -> Dict[str, int]:
        """Merge budget recommendations from all sources"""
        # Use Agent 2's budget allocation as primary (most sophisticated)
        if state["agent2_output"] and "budget_allocation" in state["agent2_output"]:
            return state["agent2_output"]["budget_allocation"]

        # Fallback to default allocation
        return {
            "email": 20000,
            "social_media": 15000,
            "display_ads": 10000,
            "influencer": 5000,
        }

    def _assess_combined_risks(self, state: WorkflowState) -> Dict[str, Any]:
        """Assess combined risks from all analyses"""
        return {
            "overall_risk": "medium",
            "key_risks": [
                "Market competition",
                "Budget constraints",
                "Timing challenges",
            ],
            "mitigation_strategies": [
                "AI-powered real-time optimization",
                "Multi-channel diversification",
                "Continuous performance monitoring",
            ],
        }

    def _generate_next_steps(self, state: WorkflowState) -> List[str]:
        """Generate next steps for campaign execution"""
        return [
            "Finalize budget allocation across recommended channels",
            "Set up AI monitoring and optimization systems",
            "Prepare campaign creative assets",
            "Configure tracking and analytics",
            "Schedule campaign launch and review milestones",
        ]

    def _simulate_human_approval(
        self, recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate human approval process"""
        # In a real system, this would involve actual human interaction
        confidence = recommendation.get("combined_confidence", 0.5)

        # Auto-approve high confidence recommendations
        if confidence > 0.8:
            return {
                "approved": True,
                "feedback": "High confidence recommendation approved",
                "modifications": {},
            }
        else:
            return {
                "approved": True,
                "feedback": "Approved with suggested monitoring",
                "modifications": {
                    "monitoring_frequency": "daily",
                    "review_checkpoints": ["week 1", "week 2", "week 4"],
                },
            }

    def _create_execution_timeline(self) -> Dict[str, str]:
        """Create execution timeline"""
        return {
            "week_1": "Campaign setup and creative preparation",
            "week_2": "Campaign launch and initial optimization",
            "week_3-4": "Performance monitoring and AI optimization",
            "week_5-6": "Results analysis and strategy refinement",
        }

    def _create_monitoring_plan(self) -> Dict[str, Any]:
        """Create monitoring plan"""
        return {
            "frequency": "real-time with daily reports",
            "key_metrics": ["ROI", "conversion_rate", "reach", "engagement"],
            "ai_optimization": "continuous",
            "human_review": "weekly",
        }

    def _define_success_metrics(self, state: WorkflowState) -> List[str]:
        """Define success metrics based on workflow results"""
        thinking_result = state.get("thinking_result", {})
        simulation = thinking_result.get("simulation", {})
        predicted_outcomes = simulation.get("predicted_outcomes", {})

        return [
            f"ROI > {predicted_outcomes.get('roi_estimate', 2.5)}",
            f"Conversion rate > {predicted_outcomes.get('conversion_rate', 0.05) * 100:.1f}%",
            f"Reach > {predicted_outcomes.get('reach_estimate', 10000):,}",
            f"Engagement rate > {predicted_outcomes.get('engagement_rate', 0.06) * 100:.1f}%",
        ]

    def run_workflow(
        self, campaign_data: Dict[str, Any], customer_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the complete LangGraph multi-agent workflow"""
        print("\n🚀 Starting LangGraph Multi-Agent Workflow")
        print("=" * 50)

        # Initialize state
        initial_state = WorkflowState(
            messages=[],
            campaign_data=campaign_data,
            customer_data=customer_data,
            agent1_output=None,
            agent2_output=None,
            reflect_agent_output=None,
            thinking_result=None,
            human_approval=None,
            final_recommendation=None,
            workflow_status="starting",
            confidence_score=0.0,
            iteration_count=0,
        )

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)

            print("\n✅ LangGraph Workflow Completed Successfully!")
            print("=" * 50)
            print(f"Final Status: {final_state['workflow_status']}")
            print(f"Final Confidence: {final_state['confidence_score']:.2f}")
            print(f"Messages Processed: {len(final_state['messages'])}")

            return final_state["final_recommendation"]

        except Exception as e:
            print(f"\n❌ LangGraph Workflow Failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "workflow_type": "LangGraph Multi-Agent",
            }


# Convenience function for easy workflow execution
def run_multi_agent_workflow(
    campaign_data: Dict[str, Any], customer_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convenience function to run the multi-agent workflow"""
    workflow = MultiAgentWorkflow()
    return workflow.run_workflow(campaign_data, customer_data)
