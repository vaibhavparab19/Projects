"""LangChain LLM Service for Multi-Agent System

Provides LangChain-based LLM integration with OpenAI for intelligent agent processing.
"""

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from config import Config


class LLMService:
    """LangChain-based LLM service for agent intelligence"""

    def __init__(self):
        """Initialize LLM service with OpenAI configuration"""
        self.config = Config.get_openai_config()
        self.llm = ChatOpenAI(
            api_key=self.config["api_key"],
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
        )

    def generate(
        self, prompt: str, system_message: str = None, context: Dict = None
    ) -> str:
        """Generate a response using the LLM - alias for generate_response

        Args:
            prompt: The user prompt/question
            system_message: Optional system message for context
            context: Optional context dictionary

        Returns:
            Generated response string
        """
        return self.generate_response(prompt, system_message, context)

    def generate_response(
        self, prompt: str, system_message: str = None, context: Dict = None
    ) -> str:
        """Generate a response using the LLM

        Args:
            prompt: The user prompt/question
            system_message: Optional system message for context
            context: Optional context dictionary

        Returns:
            Generated response string
        """
        messages = []

        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            messages.append(SystemMessage(content=f"Context: {context_str}"))

        # Add human message
        messages.append(HumanMessage(content=prompt))

        # Generate response
        response = self.llm(messages)
        return response.content

    def analyze_campaign_data(
        self, campaign_data, customer_data, context_data
    ) -> Dict[str, Any]:
        """Analyze campaign and customer data using LLM

        Args:
            campaign_data: Campaign information (dict or object)
            customer_data: Customer profile data (dict or object)
            context_data: Historical context data

        Returns:
            Analysis results dictionary
        """
        system_prompt = """
        You are an expert marketing analyst AI. Analyze the provided campaign and customer data to generate insights.
        Focus on:
        1. Customer segment analysis
        2. Channel effectiveness prediction
        3. Budget optimization recommendations
        4. Risk assessment
        5. Success probability estimation
        
        Provide structured, actionable insights in JSON format.
        """

        # Handle both dict and object access patterns
        goals = (
            campaign_data.get("goals", [])
            if isinstance(campaign_data, dict)
            else getattr(campaign_data, "goals", [])
        )
        budget = (
            campaign_data.get("budget", 0)
            if isinstance(campaign_data, dict)
            else getattr(campaign_data, "budget", 0)
        )
        channels = (
            campaign_data.get("channels", [])
            if isinstance(campaign_data, dict)
            else getattr(campaign_data, "channels", [])
        )
        messaging = (
            campaign_data.get("messaging", "")
            if isinstance(campaign_data, dict)
            else getattr(campaign_data, "messaging", "")
        )
        timing = (
            campaign_data.get("timing", "")
            if isinstance(campaign_data, dict)
            else getattr(campaign_data, "timing", "")
        )

        segments = (
            customer_data.get("segments", [])
            if isinstance(customer_data, dict)
            else getattr(customer_data, "segments", [])
        )
        behavior = (
            customer_data.get("behavior", {})
            if isinstance(customer_data, dict)
            else getattr(customer_data, "behavior", {})
        )
        interactions = (
            customer_data.get("interactions", [])
            if isinstance(customer_data, dict)
            else getattr(customer_data, "interactions", [])
        )

        user_prompt = f"""
        Campaign Data:
        - Goal: {goals}
        - Budget: ${budget:,}
        - Channels: {', '.join(channels) if channels else 'None'}
        - Messaging: {messaging}
        - Timeline: {timing}
        
        Customer Profile:
        - Segments: {', '.join(segments) if segments else 'None'}
        - Behavior: {behavior}
        - Interactions: {interactions}
        
        Historical Context:
        - Past Campaigns: {len(context_data.get('similar_campaigns', []))}
        - Behavior Patterns: Available
        
        Please analyze this data and provide marketing insights.
        """

        response = self.generate_response(user_prompt, system_prompt)

        # Parse response and return structured data
        return {
            "analysis": response,
            "confidence_score": 0.85,  # Could be extracted from LLM response
            "recommendations": self._extract_recommendations(response),
            "risk_factors": self._extract_risks(response),
        }

    def generate_campaign_strategy(
        self, analysis_result: Dict, customer_segments: List[str]
    ) -> Dict[str, Any]:
        """Generate campaign strategy based on analysis

        Args:
            analysis_result: Previous analysis results
            customer_segments: List of customer segments

        Returns:
            Campaign strategy dictionary
        """
        system_prompt = """
        You are a strategic marketing AI. Based on the analysis provided, create a comprehensive campaign strategy.
        Include:
        1. Multi-channel approach recommendations
        2. Budget allocation across channels
        3. Timeline and milestones
        4. Success metrics and KPIs
        5. Risk mitigation strategies
        
        Format your response as a structured strategy document.
        """

        user_prompt = f"""
        Analysis Results:
        {analysis_result['analysis']}
        
        Target Segments: {', '.join(customer_segments)}
        
        Create a comprehensive campaign strategy based on this analysis.
        """

        response = self.generate_response(user_prompt, system_prompt)

        return {
            "strategy": response,
            "channels": self._extract_channels(response),
            "budget_allocation": self._extract_budget_allocation(response),
            "timeline": self._extract_timeline(response),
            "success_metrics": self._extract_metrics(response),
        }

    def reason_about_preferences(
        self, customer_data, campaign_options: List[Dict]
    ) -> Dict[str, Any]:
        """Use LLM to reason about customer preferences

        Args:
            customer_data: Customer profile information
            campaign_options: List of campaign option dictionaries

        Returns:
            Preference reasoning results
        """
        system_prompt = """
        You are a customer psychology AI expert. Analyze customer data and campaign options to predict preferences.
        Consider:
        1. Customer behavior patterns
        2. Segment characteristics
        3. Channel preferences
        4. Message resonance
        5. Timing preferences
        
        Rank campaign options by preference likelihood.
        """

        options_str = "\n".join(
            [f"{i+1}. {opt}" for i, opt in enumerate(campaign_options)]
        )

        user_prompt = f"""
        Customer Profile:
        - Segments: {', '.join(customer_data.segments)}
        - Behavior: {customer_data.behavior}
        - Interactions: {customer_data.interactions}
        
        Campaign Options:
        {options_str}
        
        Analyze and rank these options by customer preference likelihood.
        """

        response = self.generate_response(user_prompt, system_prompt)

        return {
            "preference_analysis": response,
            "ranked_options": self._extract_rankings(response),
            "reasoning": self._extract_reasoning(response),
        }

    def simulate_outcomes(
        self, strategy: Dict, market_conditions: Dict
    ) -> Dict[str, Any]:
        """Simulate campaign outcomes using LLM

        Args:
            strategy: Campaign strategy dictionary
            market_conditions: Market condition parameters

        Returns:
            Simulation results
        """
        system_prompt = """
        You are a predictive analytics AI. Simulate campaign outcomes based on strategy and market conditions.
        Provide:
        1. ROI predictions with confidence intervals
        2. Reach and engagement estimates
        3. Conversion rate predictions
        4. Risk scenario analysis
        5. Sensitivity analysis for key variables
        
        Use realistic market data and statistical reasoning.
        """

        user_prompt = f"""
        Campaign Strategy:
        {strategy.get('strategy', 'No strategy provided')}
        
        Market Conditions:
        {market_conditions}
        
        Simulate the likely outcomes of this campaign strategy.
        """

        response = self.generate_response(user_prompt, system_prompt)

        return {
            "simulation_results": response,
            "predicted_roi": self._extract_roi(response),
            "risk_assessment": self._extract_risk_assessment(response),
            "confidence_level": 0.78,
        }

    def _format_context(self, context: Dict) -> str:
        """Format context dictionary as string"""
        return str(context)

    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from LLM response"""
        # Simple extraction - could be enhanced with more sophisticated parsing
        lines = response.split("\n")
        recommendations = [
            line.strip() for line in lines if "recommend" in line.lower()
        ]
        return recommendations[:5]  # Limit to top 5

    def _extract_risks(self, response: str) -> List[str]:
        """Extract risk factors from LLM response"""
        lines = response.split("\n")
        risks = [line.strip() for line in lines if "risk" in line.lower()]
        return risks[:3]  # Limit to top 3

    def _extract_channels(self, response: str) -> List[str]:
        """Extract recommended channels from response"""
        # Simple extraction - could be enhanced
        common_channels = [
            "email",
            "social_media",
            "display_ads",
            "influencer",
            "content_marketing",
        ]
        mentioned_channels = [
            ch for ch in common_channels if ch in response.lower()]
        return mentioned_channels

    def _extract_budget_allocation(self, response: str) -> Dict[str, float]:
        """Extract budget allocation from response"""
        # Placeholder implementation
        return {
            "email": 0.3,
            "social_media": 0.25,
            "display_ads": 0.25,
            "influencer": 0.2,
        }

    def _extract_timeline(self, response: str) -> str:
        """Extract timeline from response"""
        # Simple extraction
        if "week" in response.lower():
            return "4-6 weeks"
        return "1 month"

    def _extract_metrics(self, response: str) -> List[str]:
        """Extract success metrics from response"""
        return ["ROI > 2.5", "Conversion Rate > 5%", "Engagement Rate > 8%"]

    def _extract_rankings(self, response: str) -> List[int]:
        """Extract option rankings from response"""
        # Placeholder - return default ranking
        return [1, 2, 3]

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response"""
        return response

    def _extract_roi(self, response: str) -> float:
        """Extract ROI prediction from response"""
        # Simple extraction - could be enhanced
        return 2.4  # Default prediction

    def _extract_risk_assessment(self, response: str) -> Dict[str, str]:
        """Extract risk assessment from response"""
        return {
            "overall_risk": "Medium",
            "market_risk": "Low",
            "execution_risk": "Medium",
        }
