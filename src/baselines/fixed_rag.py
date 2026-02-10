from typing import Tuple

class FixedRAG:
    """Fixed retrieval baseline for comparison"""
    
    def __init__(self, config, generator, retriever, fixed_count=3):
        self.config = config
        self.generator = generator
        self.retriever = retriever
        self.fixed_count = fixed_count
        
    def answer(self, question: str) -> Tuple[str, int, float]:
        """
        Answer question with fixed retrieval count
        
        Returns:
            answer: Generated answer
            retrieval_count: Number of retrievals (always fixed_count)
            latency: Time taken
        """
        import time
        start_time = time.time()
        
        # Always retrieve fixed number
        docs = self.retriever.retrieve(question, k=self.fixed_count)
        
        # Generate answer
        answer = self.generator.generate(question, docs)
        
        latency = time.time() - start_time
        
        return answer, self.fixed_count, latency