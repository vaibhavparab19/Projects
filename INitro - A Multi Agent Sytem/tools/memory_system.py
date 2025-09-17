from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
from datetime import datetime


class MemorySystem:
    """Enhanced memory system for Agent2 with conversation memory and knowledge retrieval"""

    def __init__(self, llm, persist_directory: str = "./memory_store"):
        self.llm = llm
        self.persist_directory = persist_directory

        # Initialize conversation memory
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            return_messages=True,
            memory_key="chat_history",
        )

        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._initialize_vector_store()

        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            else:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                # Add initial knowledge base
                self._populate_initial_knowledge()
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            self.vector_store = None

    def _populate_initial_knowledge(self):
        """Populate vector store with initial marketing knowledge"""
        initial_knowledge = [
            {
                "content": "Digital marketing campaigns should focus on customer segmentation, targeting, and personalization to maximize ROI.",
                "metadata": {"type": "strategy", "topic": "segmentation"},
            },
            {
                "content": "A/B testing is crucial for optimizing campaign performance. Test different headlines, images, and call-to-actions.",
                "metadata": {"type": "optimization", "topic": "testing"},
            },
            {
                "content": "Customer lifetime value (CLV) should be considered when allocating marketing budgets across different channels.",
                "metadata": {"type": "metrics", "topic": "clv"},
            },
            {
                "content": "Social media campaigns perform best with visual content and authentic storytelling that resonates with target audiences.",
                "metadata": {"type": "channel", "topic": "social_media"},
            },
            {
                "content": "Email marketing remains one of the highest ROI channels, with personalized content driving 6x higher transaction rates.",
                "metadata": {"type": "channel", "topic": "email"},
            },
        ]

        documents = []
        for item in initial_knowledge:
            doc = Document(
                page_content=item["content"], metadata=item["metadata"])
            documents.append(doc)

        if self.vector_store and documents:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()

    def add_conversation_turn(self, human_input: str, ai_response: str):
        """Add a conversation turn to memory"""
        self.conversation_memory.chat_memory.add_user_message(human_input)
        self.conversation_memory.chat_memory.add_ai_message(ai_response)

    def get_conversation_context(self) -> str:
        """Get conversation context for current session"""
        try:
            memory_variables = self.conversation_memory.load_memory_variables({
            })
            chat_history = memory_variables.get("chat_history", [])

            # Convert message objects to strings if they exist
            if isinstance(chat_history, list):
                formatted_history = []
                for msg in chat_history:
                    if hasattr(msg, "content"):
                        role = "Human" if isinstance(
                            msg, HumanMessage) else "AI"
                        formatted_history.append(f"{role}: {msg.content}")
                    else:
                        formatted_history.append(str(msg))
                return "\n".join(formatted_history)
            else:
                return str(chat_history)
        except Exception as e:
            print(f"Warning: Could not load conversation context: {e}")
            return ""

    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add new knowledge to the vector store"""
        if not self.vector_store:
            return

        if metadata is None:
            metadata = {"timestamp": datetime.now().isoformat()}

        # Split content into chunks
        chunks = self.text_splitter.split_text(content)

        documents = []
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)

        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def retrieve_relevant_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant knowledge based on query"""
        if not self.vector_store:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Warning: Could not retrieve knowledge: {e}")
            return []

    def get_contextual_information(self, query: str) -> Dict[str, Any]:
        """Get comprehensive contextual information for query"""
        context = {
            "conversation_history": self.get_conversation_context(),
            "relevant_knowledge": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Retrieve relevant knowledge
        relevant_docs = self.retrieve_relevant_knowledge(query)
        for doc in relevant_docs:
            context["relevant_knowledge"].append(
                {"content": doc.page_content, "metadata": doc.metadata}
            )

        return context

    def clear_conversation_memory(self):
        """Clear conversation memory for new session"""
        self.conversation_memory.clear()

    def store_interaction(
        self, user_query: str, ai_response: str, session_id: str = None
    ):
        """Store a user-AI interaction"""
        # Add to conversation memory
        self.add_conversation_turn(user_query, ai_response)

        # Also add to knowledge base for future retrieval
        interaction_content = f"User Query: {user_query}\nAI Response: {ai_response}"
        metadata = {"type": "interaction",
                    "timestamp": datetime.now().isoformat()}

        if session_id:
            metadata["session_id"] = session_id

        self.add_knowledge(interaction_content, metadata)

    def save_session_summary(self, session_id: str, summary: str):
        """Save session summary for future reference"""
        session_data = {
            "session_id": session_id,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        # Add session summary to knowledge base
        self.add_knowledge(
            content=f"Session Summary: {summary}",
            metadata={
                "type": "session_summary",
                "session_id": session_id,
                "timestamp": session_data["timestamp"],
            },
        )
