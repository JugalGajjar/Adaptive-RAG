"""
Quick Data Validation Script for Adaptive RAG
to verify the data files are correctly formatted.
"""

import json
from pathlib import Path
from collections import defaultdict

def validate_data():
    """Validate data files"""
    
    print("\n" + "="*60)
    print("ADAPTIVE RAG DATA VALIDATION")
    print("="*60)
    
    data_dir = Path("data")
    issues = []
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\nERROR: 'data/' directory not found")
        print(f"   Please create it and add your .jsonl files")
        return
    
    # Check required files
    required_files = {
        "train": data_dir / "train.jsonl",
        "val": data_dir / "val.jsonl",
        "test": data_dir / "test.jsonl",
        "corpus": data_dir / "corpus" / "corpus.jsonl"
    }
    
    print("\nChecking files...")
    for name, path in required_files.items():
        if path.exists():
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: {path} (NOT FOUND)")
            issues.append(f"Missing: {path}")
    
    if issues:
        print(f"\nCannot proceed - missing files")
        return
    
    # Validate each file
    print("\n" + "="*60)
    print("VALIDATING CONTENT")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")
        path = required_files[split]
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            
            print(f"  Total: {len(lines)} questions")
            
            # Check fields
            difficulty_counts = defaultdict(int)
            with_context = 0
            
            for i, item in enumerate(lines, 1):
                required = ["id", "question", "answer", "difficulty"]
                missing = [f for f in required if f not in item]
                
                if missing:
                    print(f"  Line {i}: Missing {missing}")
                    issues.append(f"{split} line {i}: missing {missing}")
                
                diff = item.get("difficulty", "unknown")
                difficulty_counts[diff] += 1
                
                if diff not in ["simple", "medium", "hard"]:
                    print(f"  Line {i}: Invalid difficulty '{diff}'")
                
                if "context" in item and item["context"]:
                    with_context += 1
            
            print(f"  With context: {with_context}/{len(lines)} ({with_context/len(lines)*100:.0f}%)")
            print(f"  Difficulty:")
            for diff in ["simple", "medium", "hard"]:
                count = difficulty_counts[diff]
                pct = count/len(lines)*100 if lines else 0
                print(f"    {diff}: {count} ({pct:.0f}%)")
            
        except Exception as e:
            print(f"  Error: {e}")
            issues.append(f"{split}: {e}")
    
    # Validate corpus
    print(f"\nCORPUS")
    corpus_path = required_files["corpus"]
    
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]
        
        print(f"  Total: {len(corpus)} documents")
        
        avg_length = sum(len(doc.get("text", "")) for doc in corpus) / len(corpus) if corpus else 0
        print(f"  Avg length: {avg_length:.0f} chars")
        
    except Exception as e:
        print(f"  Error: {e}")
        issues.append(f"corpus: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if issues:
        print(f"\nFound {len(issues)} issues")
        print("\nPlease fix these before training.")
    else:
        print("\nAll data files are valid!")
        print("\nYou're ready to train:")
        print("  python train.py")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    validate_data()