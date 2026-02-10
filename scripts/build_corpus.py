import json
import jsonlines
from pathlib import Path

def create_corpus_from_datasets():
    """Extract unique documents from train/val/test contexts"""
    
    all_docs = {}  # Use dict to deduplicate by content
    doc_id = 1
    
    # Load all datasets
    for split in ["train", "val", "test"]:
        filepath = Path(f"data/{split}.jsonl")
        
        with jsonlines.open(filepath) as reader:
            for item in reader:
                # Extract context documents
                contexts = item.get("context", [])
                
                for doc_text in contexts:
                    # Deduplicate by text content
                    if doc_text not in all_docs:
                        all_docs[doc_text] = {
                            "id": f"doc_{doc_id}",
                            "text": doc_text,
                            "metadata": {
                                "source_question_id": item.get("id"),
                                "difficulty": item.get("difficulty")
                            }
                        }
                        doc_id += 1
        
        print(f"Processed {split} set, total unique docs so far: {len(all_docs)}")
    
    # Save corpus
    corpus_path = Path("data/corpus/corpus.jsonl")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(corpus_path, "w") as writer:
        for doc in all_docs.values():
            writer.write(doc)
    
    print(f"Created corpus with {len(all_docs)} unique documents")
    print(f"   Saved to: {corpus_path}")
    
    return list(all_docs.values())

# Run this to create your corpus
corpus = create_corpus_from_datasets()