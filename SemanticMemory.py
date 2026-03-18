import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from datetime import datetime

class SimpleMemorySystem:
    """
    A basic implementation of semantic memory for LLMs using vector embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize the embedding model (converts text to vectors)
        self.embedding_model = SentenceTransformer(model_name)
        
        # Memory storage: list of dicts with text, embedding, and metadata
        self.memories: List[Dict[str, Any]] = []
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Convert text to a vector embedding."""
        return self.embedding_model.encode(text)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def store_memory(self, 
                     text: str, 
                     memory_type: str = "conversation", 
                     importance: float = 1.0,
                     metadata: Dict = None) -> None:
        """
        Store a memory by converting it to an embedding and saving it.
        
        Args:
            text: The content to remember
            memory_type: e.g., 'fact', 'preference', 'conversation'
            importance: How important this memory is (0-1)
            metadata: Additional info like timestamp, topic, etc.
        """
        # Generate embedding for the text
        embedding = self._generate_embedding(text)
        
        # Create memory entry
        memory = {
            'id': len(self.memories),
            'text': text,
            'embedding': embedding,
            'type': memory_type,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.memories.append(memory)
        print(f"✅ Stored memory: '{text[:50]}...'")
        
    def retrieve_relevant_memories(self, 
                                   query: str, 
                                   top_k: int = 3,
                                   min_similarity: float = 0.3,
                                   memory_types: List[str] = None) -> List[Dict]:
        """
        Retrieve memories relevant to a query using semantic search.
        
        Args:
            query: The text to search for
            top_k: Maximum number of memories to return
            min_similarity: Minimum similarity threshold
            memory_types: Optional filter by memory type
        """
        if not self.memories:
            return []
            
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            # Apply type filter if specified
            if memory_types and memory['type'] not in memory_types:
                continue
                
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, memory['embedding'])
            
            # Apply importance boost (optional)
            weighted_similarity = similarity * memory['importance']
            
            similarities.append({
                'memory': memory,
                'similarity': similarity,
                'weighted_score': weighted_similarity
            })
        
        # Sort by weighted score and filter by threshold
        relevant = [s for s in similarities if s['similarity'] >= min_similarity]
        relevant.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return relevant[:top_k]
    
    def build_augmented_prompt(self, query: str, current_context: str = "") -> str:
        """
        Build a prompt augmented with relevant memories.
        This simulates what happens before sending to an LLM.
        """
        # Retrieve relevant memories
        memories = self.retrieve_relevant_memories(query)
        
        if not memories:
            return current_context + "\n\nUser: " + query
        
        # Format memories for the prompt
        memory_text = "Relevant past memories:\n"
        for i, m in enumerate(memories, 1):
            memory = m['memory']
            memory_text += f"{i}. [{memory['type']}] {memory['text']}\n"
        
        # Build augmented prompt
        augmented_prompt = f"""
{memory_text}

Current context:
{current_context}

User query: {query}

Please respond considering both the current query and the relevant past memories above.
"""
        return augmented_prompt

# ============================================
# DEMONSTRATION
# ============================================

def demonstrate_memory_system():
    print("="*60)
    print("LLM MEMORY SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize the memory system
    memory_system = SimpleMemorySystem()
    
    # Simulate a multi-session conversation history
    print("\n📝 SESSION 1: Storing memories...")
    
    # Store various types of memories
    memory_system.store_memory(
        "The user loves Italian cuisine and especially enjoys pasta carbonara",
        memory_type="preference",
        importance=0.9,
        metadata={"topic": "food"}
    )
    
    memory_system.store_memory(
        "User works as a software engineer specializing in Python and machine learning",
        memory_type="fact",
        importance=0.8,
        metadata={"topic": "career"}
    )
    
    memory_system.store_memory(
        "Discussed the book 'Dune' and user mentioned they prefer Frank Herbert's original series",
        memory_type="conversation",
        importance=0.6,
        metadata={"topic": "books"}
    )
    
    memory_system.store_memory(
        "User has a dog named Max, a golden retriever who is 3 years old",
        memory_type="fact",
        importance=0.7,
        metadata={"topic": "pets"}
    )
    
    memory_system.store_memory(
        "User is planning a trip to Japan next spring and wants restaurant recommendations",
        memory_type="conversation",
        importance=0.8,
        metadata={"topic": "travel"}
    )
    
    print("\n🔍 SESSION 2: Retrieving memories based on new queries...")
    
    # Test queries to demonstrate semantic search
    test_queries = [
        "What should I eat for dinner?",
        "Can you help me debug my Python code?",
        "I'm looking for a good science fiction book",
        "Tell me about your favorite pets",
        "What's the best time to visit Tokyo?"
    ]
    
    for query in test_queries:
        print(f"\n📌 Query: '{query}'")
        print("-" * 40)
        
        # Retrieve relevant memories
        memories = memory_system.retrieve_relevant_memories(query, top_k=2)
        
        if memories:
            print("Retrieved relevant memories:")
            for i, m in enumerate(memories, 1):
                memory = m['memory']
                print(f"  {i}. [{memory['type']}] {memory['text']} (similarity: {m['similarity']:.3f})")
            
            # Show the augmented prompt that would be sent to an LLM
            print("\nAugmented prompt (simulated):")
            print(memory_system.build_augmented_prompt(query))
        else:
            print("No relevant memories found.")
    
    # Demonstrate memory consolidation/extraction
    print("\n🔄 SESSION 3: Extracting and storing new memories from conversation")
    
    # Simulate an LLM response
    conversation = """
    User: I'm thinking of learning a new programming language.
    LLM: That's great! You mentioned you're a Python developer. 
         Since you enjoy Python and machine learning, you might like Julia for scientific computing.
    """
    
    # In a real system, you'd analyze the conversation to extract key info
    # Here we manually extract what might be stored
    new_memory = "User is interested in learning Julia programming language for scientific computing"
    memory_system.store_memory(
        new_memory,
        memory_type="preference",
        importance=0.5,
        metadata={"topic": "programming", "source": "conversation"}
    )
    
    # Test retrieval with the new memory
    print("\n📌 Testing new memory:")
    query = "What programming languages should I learn next?"
    memories = memory_system.retrieve_relevant_memories(query, top_k=1)
    if memories:
        print(f"Retrieved: {memories[0]['memory']['text']}")

if __name__ == "__main__":
    demonstrate_memory_system()
