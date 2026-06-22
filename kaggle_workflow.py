#!/usr/bin/env python3
"""
kaggle_workflow.py - Complete training and testing workflow for Kaggle GPU
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run command and print status"""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    UrduFinance Kaggle GPU Workflow                       ║
║                     Train → Build → Test                             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU
    run_command("nvidia-smi", "GPU Check")
    
    steps = [
        # 1. Install dependencies
        ("pip install -q -r requirements.txt", "Installing dependencies"),
        
        # 2. Check environment
        ("python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'", "Check PyTorch CUDA"),
        
        # 3. Download dataset from HF (if not local)
        ("python -c \"from datasets import load_dataset; ds = load_dataset('hassan7272/urdu-finance-qa', split='train'); print(f'Downloaded {len(ds)} samples')\"", "Download dataset from HF"),
        
        # 4. Build corpus and indexes (CRITICAL - this creates the FAISS indexes)
        ("python retrieval/build_corpus.py --source hf --dataset hassan7272/urdu-finance-qa --output-dir retrieval/artifacts", "Build corpus and FAISS indexes"),
        
        # 5. Build BM25 index
        ("python -c \"import json; from retrieval.bm25_retriever import build_bm25_index; build_bm25_index('data/processed/qa_corpus.jsonl', 'retrieval/artifacts/bm25_index.pkl')\"", "Build BM25 index"),
        
        # 6. Run unit tests
        ("pytest tests/test_cache.py tests/test_llm_client.py -v --tb=short -x", "Unit tests"),
        
        # 7. Run real tests with your API
        ("python run_real_tests.py", "Real integration tests"),
        
        # 8. Coverage report
        ("pytest tests/ --cov=finguard --cov-report=term-missing --cov-fail-under=50", "Coverage report"),
    ]
    
    results = {}
    for cmd, desc in steps:
        results[desc] = run_command(cmd, desc)
    
    # Summary
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    for desc, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️ SOME STEPS FAILED - CHECK OUTPUT ABOVE")
    print("="*70)

if __name__ == "__main__":
    main()
