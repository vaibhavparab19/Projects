from typing import List, Dict, Any, Optional
from langchain.retrievers import MultiVectorRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from datetime import datetime


class RetrievalSystem:
    """Advanced retrieval system for contextual knowledge access"""

    def __init__(self, llm, vector_store: Chroma):
        self.llm = llm
        self.vector_store = vector_store
        self.embeddings = OpenAIEmbeddings()

        # Initialize retrievers
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        # Contextual compression retriever for better relevance
        self.compressor = LLMChainExtractor.from_llm(llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.base_retriever
        )

    def retrieve_campaign_insights(
        self, campaign_data: Dict[str, Any]
    ) -> List[Document]:
        """Retrieve insights specific to campaign characteristics"""
        # Build query from campaign data
        query_parts = []

        if "budget" in campaign_data:
            budget_range = self._categorize_budget(campaign_data["budget"])
            query_parts.append(f"{budget_range} budget campaigns")

        if "channels" in campaign_data:
            channels = ", ".join(campaign_data["channels"])
            query_parts.append(f"{channels} marketing channels")

        if "target_audience" in campaign_data:
            audience = campaign_data["target_audience"]
            query_parts.append(f"{audience} target audience")

        query = (
            " ".join(
                query_parts) if query_parts else "marketing campaign optimization"
        )

        try:
            docs = self.compression_retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve campaign insights: {e}")
            return self.base_retriever.get_relevant_documents(query)

    def retrieve_customer_insights(
        self, customer_data: Dict[str, Any]
    ) -> List[Document]:
        """Retrieve insights based on customer characteristics"""
        query_parts = []

        if hasattr(customer_data, "behavior") and customer_data.behavior:
            behaviors = ", ".join(customer_data.behavior)
            query_parts.append(f"customer behavior {behaviors}")

        if hasattr(customer_data, "preferences") and customer_data.preferences:
            prefs = ", ".join(customer_data.preferences)
            query_parts.append(f"customer preferences {prefs}")

        if hasattr(customer_data, "demographics") and customer_data.demographics:
            demo = customer_data.demographics
            if "age_group" in demo:
                query_parts.append(f"{demo['age_group']} age group")
            if "location" in demo:
                query_parts.append(f"{demo['location']} market")

        query = (
            " ".join(
                query_parts) if query_parts else "customer segmentation targeting"
        )

        try:
            docs = self.compression_retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve customer insights: {e}")
            return self.base_retriever.get_relevant_documents(query)

    def retrieve_contextual_knowledge(self, context: str, query: str) -> List[Document]:
        """Retrieve knowledge based on context and specific query"""
        enhanced_query = f"{context} {query}"

        try:
            docs = self.compression_retriever.get_relevant_documents(
                enhanced_query)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve contextual knowledge: {e}")
            return self.base_retriever.get_relevant_documents(query)

    def retrieve_similar_campaigns(
        self, campaign_features: Dict[str, Any]
    ) -> List[Document]:
        """Retrieve information about similar successful campaigns"""
        # Build similarity query
        query_parts = ["successful campaign case study"]

        if "industry" in campaign_features:
            query_parts.append(f"{campaign_features['industry']} industry")

        if "campaign_type" in campaign_features:
            query_parts.append(
                f"{campaign_features['campaign_type']} campaign")

        if "objectives" in campaign_features:
            objectives = ", ".join(campaign_features["objectives"])
            query_parts.append(f"{objectives} objectives")

        query = " ".join(query_parts)

        try:
            docs = self.base_retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve similar campaigns: {e}")
            return []

    def retrieve_best_practices(self, domain: str) -> List[Document]:
        """Retrieve best practices for specific marketing domain"""
        query = f"{domain} marketing best practices optimization"

        try:
            docs = self.base_retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve best practices: {e}")
            return []

    def _categorize_budget(self, budget: float) -> str:
        """Categorize budget into ranges"""
        if budget < 1000:
            return "low"
        elif budget < 10000:
            return "medium"
        elif budget < 100000:
            return "high"
        else:
            return "enterprise"

    def add_campaign_result(
        self, campaign_data: Dict[str, Any], results: Dict[str, Any]
    ):
        """Add campaign results to knowledge base for future retrieval"""
        # Create document from campaign and results
        content = self._format_campaign_result(campaign_data, results)

        metadata = {
            "type": "campaign_result",
            "timestamp": datetime.now().isoformat(),
            "budget": campaign_data.get("budget", 0),
            "channels": campaign_data.get("channels", []),
            "roi": results.get("predicted_roi", 0),
            "confidence": results.get("confidence_score", 0),
        }

        doc = Document(page_content=content, metadata=metadata)

        try:
            self.vector_store.add_documents([doc])
            self.vector_store.persist()
        except Exception as e:
            print(f"Warning: Could not add campaign result: {e}")

    def _format_campaign_result(
        self, campaign_data: Dict[str, Any], results: Dict[str, Any]
    ) -> str:
        """Format campaign and results into searchable text"""
        content_parts = []

        # Campaign details
        content_parts.append(
            f"Campaign Budget: ${campaign_data.get('budget', 0)}")

        if "channels" in campaign_data:
            channels = ", ".join(campaign_data["channels"])
            content_parts.append(f"Channels: {channels}")

        if "target_audience" in campaign_data:
            content_parts.append(
                f"Target Audience: {campaign_data['target_audience']}")

        # Results
        if "predicted_roi" in results:
            content_parts.append(f"Achieved ROI: {results['predicted_roi']}x")

        if "confidence_score" in results:
            content_parts.append(
                f"Confidence Score: {results['confidence_score']}%")

        if "insights" in results:
            insights = results["insights"]
            if isinstance(insights, list):
                for insight in insights:
                    if isinstance(insight, dict) and "insight" in insight:
                        content_parts.append(f"Insight: {insight['insight']}")

        return "\n".join(content_parts)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        try:
            # Get collection info if available
            collection = self.vector_store._collection
            count = collection.count() if hasattr(collection, "count") else 0

            return {
                "total_documents": count,
                "retriever_type": "contextual_compression",
                "embedding_model": "openai",
                "status": "active",
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "retriever_type": "basic",
                "embedding_model": "openai",
                "status": "error",
                "error": str(e),
            }
