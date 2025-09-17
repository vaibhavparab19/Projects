"""Reflect Agent - Self-Reflecting Chatbot for AI Decisioning System

This agent implements a self-reflection mechanism that evaluates and refines responses
before presenting them to users. It integrates with the existing LangGraph workflow
and provides conversational interface capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from dataclasses import dataclass
import json
import time
from datetime import datetime
import uuid

# Import existing components
from tools.memory_system import MemorySystem
from tools.llm_service import LLMService
from thinking_node.core import ThinkingLayer
from config import Config


@dataclass
class ReflectionResult:
    """Result from self-reflection process"""

    original_response: str
    reflection_score: float
    improvements: List[str]
    refined_response: str
    confidence: float
    reasoning: str


@dataclass
class ChatMessage:
    """Chat message structure"""

    id: str
    content: str
    role: str  # 'user', 'assistant', 'system'
    timestamp: datetime
    metadata: Dict[str, Any] = None


class ReflectAgent:
    """Self-Reflecting Chatbot Agent for AI Decisioning System"""

    def __init__(self):
        """Initialize the Reflect Agent with all necessary components"""
        self.config = Config()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            api_key=self.config.OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000,
        )

        # Initialize components
        self.memory_system = MemorySystem(self.llm)
        self.llm_service = LLMService()
        self.thinking_layer = ThinkingLayer()

        # Conversation memory
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, max_token_limit=4000
        )

        # Session management
        self.current_session_id = None
        self.chat_history = []

        # Reflection prompts
        self.reflection_prompt = self._create_reflection_prompt()
        self.refinement_prompt = self._create_refinement_prompt()

    def _create_reflection_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for self-reflection"""
        system_template = """You are a self-reflection module for an AI assistant. Your job is to critically evaluate responses before they are sent to users.
        
Evaluate the response on these criteria:
1. Accuracy: Is the information correct and relevant to the context?
2. Clarity: Is the response clear and easy to understand?
3. Usefulness: Does it provide actionable insights or helpful information?
4. Creativity: Are there innovative suggestions or unique perspectives?
5. Completeness: Does it fully address the user's question?

Provide a score from 0-10 and specific improvement suggestions.
        
Context: {context}
User Query: {user_query}
Original Response: {original_response}

Provide your evaluation in JSON format:
{{
    "accuracy_score": 0-10,
    "clarity_score": 0-10,
    "usefulness_score": 0-10,
    "creativity_score": 0-10,
    "completeness_score": 0-10,
    "overall_score": 0-10,
    "improvements": ["specific improvement 1", "specific improvement 2"],
    "reasoning": "detailed explanation of the evaluation"
}}"""

        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(system_template)]
        )

    def _create_refinement_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for response refinement"""
        system_template = """You are a response refinement module. Based on the reflection feedback, improve the original response.
        
Original Response: {original_response}
Reflection Feedback: {reflection_feedback}
User Query: {user_query}
Context: {context}

Create an improved response that addresses the feedback while maintaining the helpful and conversational tone.
Focus on:
- Fixing any inaccuracies
- Improving clarity and structure
- Adding creative insights or suggestions
- Ensuring completeness

Provide only the refined response, no meta-commentary."""

        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(system_template)]
        )

    def start_session(self, session_id: str = None) -> str:
        """Start a new chat session"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.current_session_id = session_id
        self.chat_history = []
        self.conversation_memory.clear()

        # Add welcome message
        welcome_msg = ChatMessage(
            id=str(uuid.uuid4()),
            content="Hello! I'm your INitro Agent. I can help you analyze campaigns, generate insights, and provide strategic recommendations. How can I assist you today?",
            role="assistant",
            timestamp=datetime.now(),
            metadata={"type": "welcome", "session_start": True},
        )
        self.chat_history.append(welcome_msg)

        return session_id

    def process_message(
        self, user_message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process user message with self-reflection"""
        try:
            # Create user message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                content=user_message,
                role="user",
                timestamp=datetime.now(),
            )
            self.chat_history.append(user_msg)

            # Add to conversation memory
            self.conversation_memory.chat_memory.add_user_message(user_message)

            # Get relevant context from memory
            memory_context = self.memory_system.get_contextual_information(
                user_message)

            # Combine contexts
            # Convert conversation buffer to string format to avoid serialization issues
            conversation_history = ""
            if (
                hasattr(self.conversation_memory, "buffer")
                and self.conversation_memory.buffer
            ):
                try:
                    conversation_history = str(self.conversation_memory.buffer)
                except:
                    conversation_history = "Previous conversation context available"

            full_context = {
                "conversation_history": conversation_history,
                "memory_context": memory_context,
                "system_context": context or {},
                "session_id": self.current_session_id,
            }

            # Generate initial response
            initial_response = self._generate_initial_response(
                user_message, full_context
            )

            # Self-reflect on the response
            reflection_result = self._reflect_on_response(
                user_message, initial_response, full_context
            )

            # Create final response message
            final_msg = ChatMessage(
                id=str(uuid.uuid4()),
                content=reflection_result.refined_response,
                role="assistant",
                timestamp=datetime.now(),
                metadata={
                    "reflection_score": reflection_result.reflection_score,
                    "confidence": reflection_result.confidence,
                    "improvements_made": len(reflection_result.improvements),
                    "original_response": reflection_result.original_response,
                },
            )
            self.chat_history.append(final_msg)

            # Add to conversation memory
            self.conversation_memory.chat_memory.add_ai_message(
                reflection_result.refined_response
            )

            # Store interaction in long-term memory
            self.memory_system.store_interaction(
                user_message,
                reflection_result.refined_response,
                self.current_session_id,
            )

            # Generate suggestions if appropriate
            suggestions = self._generate_suggestions(
                user_message, reflection_result.refined_response, full_context
            )

            return {
                "message": {
                    "id": final_msg.id,
                    "content": final_msg.content,
                    "role": final_msg.role,
                    "timestamp": final_msg.timestamp.isoformat(),
                    "metadata": final_msg.metadata or {},
                },
                "reflection": {
                    "score": reflection_result.reflection_score,
                    "confidence": reflection_result.confidence,
                    "improvements": reflection_result.improvements,
                    "reasoning": reflection_result.reasoning,
                },
                "suggestions": suggestions,
                "context_used": memory_context,
                "session_id": self.current_session_id,
            }

        except Exception as e:
            error_msg = ChatMessage(
                id=str(uuid.uuid4()),
                content=f"I apologize, but I encountered an error processing your message. Please try again.",
                role="assistant",
                timestamp=datetime.now(),
                metadata={"error": str(e), "type": "error"},
            )
            self.chat_history.append(error_msg)

            return {
                "message": {
                    "id": error_msg.id,
                    "content": error_msg.content,
                    "role": error_msg.role,
                    "timestamp": error_msg.timestamp.isoformat(),
                    "metadata": error_msg.metadata or {},
                },
                "error": str(e),
                "session_id": self.current_session_id,
            }

    def _generate_initial_response(
        self, user_message: str, context: Dict[str, Any]
    ) -> str:
        """Generate initial response using LLM service"""
        system_message = """You are INitro Agent, an AI Marketing Intelligence assistant specializing in campaign analysis, customer insights, and strategic recommendations. 
        
You have access to:
- Campaign data and performance metrics
- Customer segmentation and behavior data
- Marketing strategy frameworks
- Industry best practices
        
Provide helpful, actionable, and insightful responses. Be conversational but professional.
        
Context: {context}
        
User Query: {user_message}"""

        return self.llm_service.generate_response(
            user_message,
            system_message.format(
                context=json.dumps(context, indent=2), user_message=user_message
            ),
            context,
        )

    def _reflect_on_response(
        self, user_query: str, original_response: str, context: Dict[str, Any]
    ) -> ReflectionResult:
        """Perform self-reflection on the generated response"""
        try:
            # Get reflection evaluation
            reflection_chain = LLMChain(
                llm=self.llm, prompt=self.reflection_prompt)
            reflection_output = reflection_chain.run(
                context=json.dumps(context, indent=2),
                user_query=user_query,
                original_response=original_response,
            )

            # Parse reflection result
            try:
                reflection_data = json.loads(reflection_output)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                reflection_data = {
                    "overall_score": 7.0,
                    "improvements": ["Could not parse reflection feedback"],
                    "reasoning": "Reflection parsing failed, using original response",
                }

            overall_score = reflection_data.get("overall_score", 7.0)
            improvements = reflection_data.get("improvements", [])
            reasoning = reflection_data.get("reasoning", "")

            # If score is high enough, use original response
            if overall_score >= 8.0:
                refined_response = original_response
                confidence = 0.9
            else:
                # Refine the response
                refined_response = self._refine_response(
                    user_query, original_response, reflection_data, context
                )
                confidence = min(0.8, overall_score / 10.0 + 0.3)

            return ReflectionResult(
                original_response=original_response,
                reflection_score=overall_score,
                improvements=improvements,
                refined_response=refined_response,
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            # Fallback to original response if reflection fails
            return ReflectionResult(
                original_response=original_response,
                reflection_score=7.0,
                improvements=[f"Reflection failed: {str(e)}"],
                refined_response=original_response,
                confidence=0.7,
                reasoning="Reflection process encountered an error",
            )

    def _refine_response(
        self,
        user_query: str,
        original_response: str,
        reflection_data: Dict,
        context: Dict[str, Any],
    ) -> str:
        """Refine the response based on reflection feedback"""
        try:
            refinement_chain = LLMChain(
                llm=self.llm, prompt=self.refinement_prompt)
            refined_response = refinement_chain.run(
                original_response=original_response,
                reflection_feedback=json.dumps(reflection_data, indent=2),
                user_query=user_query,
                context=json.dumps(context, indent=2),
            )
            return refined_response.strip()
        except Exception as e:
            # Return original response if refinement fails
            return original_response

    def _generate_suggestions(
        self, user_message: str, response: str, context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate contextual suggestions for the user"""
        try:
            suggestion_prompt = f"""Based on this conversation, suggest 3 helpful follow-up questions or actions the user might want to take.
            
User Query: {user_message}
Assistant Response: {response}
            
Provide suggestions as a JSON array of objects with 'title' and 'description' fields.
Example: [{"title": "Analyze Campaign Performance", "description": "Get detailed metrics for your current campaigns"}]
            
Focus on actionable marketing intelligence tasks."""

            suggestions_response = self.llm.invoke(
                [HumanMessage(content=suggestion_prompt)]
            )

            try:
                suggestions = json.loads(suggestions_response.content)
                return suggestions[:3]  # Limit to 3 suggestions
            except json.JSONDecodeError:
                # Fallback suggestions
                return [
                    {
                        "title": "Campaign Analysis",
                        "description": "Analyze your campaign performance metrics",
                    },
                    {
                        "title": "Customer Insights",
                        "description": "Get detailed customer segmentation analysis",
                    },
                    {
                        "title": "Strategy Optimization",
                        "description": "Optimize your marketing strategy based on data",
                    },
                ]
        except Exception:
            return []

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get formatted chat history"""
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata or {},
            }
            for msg in self.chat_history
        ]

    def get_memory_context(self, query: str = None) -> Dict[str, Any]:
        """Get relevant memory context"""
        if query:
            return self.memory_system.get_contextual_information(query)
        else:
            return {
                "conversation_summary": self.conversation_memory.buffer,
                "session_id": self.current_session_id,
                "message_count": len(self.chat_history),
            }

    def trigger_system_action(
        self, action: str, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Trigger system actions like re-analysis, optimization, etc."""
        try:
            if action == "analyze_campaign":
                # Trigger campaign analysis
                return self._trigger_campaign_analysis(parameters)
            elif action == "optimize_strategy":
                # Trigger strategy optimization
                return self._trigger_strategy_optimization(parameters)
            elif action == "export_report":
                # Trigger report export
                return self._trigger_report_export(parameters)
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    def process_workflow_context(
        self, context: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Process workflow context for LangGraph integration"""
        try:
            # Create a temporary session for workflow processing
            workflow_session_id = f"workflow_{int(time.time())}"
            self.start_session(workflow_session_id)

            # Prepare workflow analysis prompt
            workflow_prompt = f"""
            You are analyzing a multi-agent workflow state. Provide conversational insights and reflections.
            
            Workflow Context:
            - Campaign Data: {json.dumps(context.get('campaign_data', {}), indent=2)}
            - Agent 1 Output: {json.dumps(context.get('agent1_output', {}), indent=2)}
            - Agent 2 Output: {json.dumps(context.get('agent2_output', {}), indent=2)}
            - Thinking Guidance: {json.dumps(context.get('thinking_guidance', {}), indent=2)}
            - Workflow Status: {context.get('workflow_status', 'unknown')}
            
            Query: {query}
            
            Provide:
            1. Analysis of the current workflow state
            2. Self-reflection on the quality and coherence of outputs
            3. Suggestions for improvement or next steps
            4. Conversational summary for user interaction
            """

            # Process through reflection pipeline
            response = self.process_message(workflow_prompt, context)

            # Extract structured insights
            result = {
                "analysis": response.get("message", {}).get("content", ""),
                "reflection": response.get("reflection", {}).get("reasoning", ""),
                "suggestions": self._extract_suggestions(
                    response.get("message", {}).get("content", "")
                ),
                "confidence": response.get("reflection", {}).get("confidence", 0.8),
                "conversational_summary": self._create_conversational_summary(
                    context, response
                ),
            }

            # Clean up temporary session
            self.clear_session()

            return result

        except Exception as e:
            return {
                "error": str(e),
                "analysis": "Error occurred during workflow analysis",
                "reflection": "Unable to perform self-reflection",
                "suggestions": [],
                "confidence": 0.0,
                "conversational_summary": "Workflow analysis failed",
            }

    def _extract_suggestions(self, response_text: str) -> List[str]:
        """Extract actionable suggestions from response text"""
        suggestions = []
        lines = response_text.split("\n")

        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["suggest", "recommend", "should", "could", "improve"]
            ):
                if len(line) > 10:  # Filter out very short lines
                    suggestions.append(line)

        return suggestions[:5]  # Limit to top 5 suggestions

    def _create_conversational_summary(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ) -> str:
        """Create a conversational summary for user interaction"""
        try:
            workflow_status = context.get("workflow_status", "unknown")
            confidence = response.get("reflection", {}).get("confidence", 0.8)

            summary = f"The workflow is currently in '{workflow_status}' status. "
            summary += f"Based on my analysis, I have {confidence:.0%} confidence in the current direction. "

            if context.get("agent1_output") and context.get("agent2_output"):
                summary += "Both agents have provided their insights, and I've reflected on their outputs. "

            suggestions = self._extract_suggestions(
                response.get("message", {}).get("content", "")
            )
            if suggestions:
                summary += f"I have {len(suggestions)} key suggestions for improvement."

            return summary

        except Exception:
            return "Workflow analysis completed with conversational insights available."

    def _trigger_campaign_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger campaign analysis using existing agents"""
        # This would integrate with the existing Agent1 and Agent2
        return {
            "status": "triggered",
            "action": "campaign_analysis",
            "message": "Campaign analysis has been initiated. Results will be available shortly.",
        }

    def _trigger_strategy_optimization(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger strategy optimization"""
        return {
            "status": "triggered",
            "action": "strategy_optimization",
            "message": "Strategy optimization process has been started.",
        }

    def _trigger_report_export(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger report export"""
        return {
            "status": "triggered",
            "action": "report_export",
            "message": "Report export has been queued for processing.",
        }

    def clear_session(self):
        """Clear current session data"""
        self.chat_history = []
        self.conversation_memory.clear()
        self.current_session_id = None

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if not self.chat_history:
            return {"message_count": 0, "session_active": False}

        user_messages = [
            msg for msg in self.chat_history if msg.role == "user"]
        assistant_messages = [
            msg for msg in self.chat_history if msg.role == "assistant"
        ]

        return {
            "session_id": self.current_session_id,
            "message_count": len(self.chat_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "session_start": (
                self.chat_history[0].timestamp.isoformat()
                if self.chat_history
                else None
            ),
            "last_activity": (
                self.chat_history[-1].timestamp.isoformat()
                if self.chat_history
                else None
            ),
            "session_active": True,
        }
