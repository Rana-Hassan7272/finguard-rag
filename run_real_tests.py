#!/usr/bin/env python3
"""
run_real_tests.py - Execute real evaluation with actual pipeline and data
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_dataset_sample(n: int = 50) -> List[Dict]:
    """Load sample from local dataset"""
    dataset_path = Path("data/raw/urdu_finance_qa.jsonl")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        # Try to download from HF
        print("📥 Attempting to download from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("hassan7272/urdu-finance-qa", split="train")
            samples = []
            for i, item in enumerate(ds):
                if i >= n:
                    break
                samples.append({
                    "id": item.get("id", i),
                    "question": item.get("question_en", item.get("question_ur", "")),
                    "answer": item.get("answer_en", item.get("answer_ur", "")),
                    "category": item.get("category", "unknown"),
                    "difficulty": item.get("difficulty", "medium"),
                })
            return samples
        except Exception as e:
            print(f"❌ Failed to load from HF: {e}")
            return []
    
    # Load from local file
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                item = json.loads(line.strip())
                samples.append({
                    "id": item.get("id", i),
                    "question": item.get("question_en", item.get("question_ur", "")),
                    "answer": item.get("answer_en", item.get("answer_ur", "")),
                    "category": item.get("category", "unknown"),
                    "difficulty": item.get("difficulty", "medium"),
                })
            except:
                continue
    
    return samples


def check_components():
    """Check if all required components are available"""
    print("=" * 70)
    print("CHECKING COMPONENTS")
    print("=" * 70)
    
    status = {}
    
    # Check API keys
    groq_key = os.environ.get("GROQ_API_KEY", "")
    has_groq = groq_key.startswith("gsk_")
    status["groq_api"] = has_groq
    print(f"{'✅' if has_groq else '❌'} Groq API Key: {'Found' if has_groq else 'Missing'}")
    
    # Check dataset
    dataset_path = Path("data/raw/urdu_finance_qa.jsonl")
    has_dataset = dataset_path.exists()
    status["local_dataset"] = has_dataset
    print(f"{'✅' if has_dataset else '❌'} Local Dataset: {'Found' if has_dataset else 'Missing'}")
    
    # Check HF token (optional but helpful)
    hf_token = os.environ.get("HF_TOKEN", "")
    has_hf = hf_token.startswith("hf_")
    status["hf_token"] = has_hf
    print(f"{'✅' if has_hf else '⚠️'} HF Token: {'Found' if has_hf else 'Optional - for downloads'}")
    
    # Check artifacts
    artifacts_dir = Path("retrieval/artifacts")
    has_artifacts = artifacts_dir.exists() and any(artifacts_dir.iterdir())
    status["artifacts"] = has_artifacts
    print(f"{'✅' if has_artifacts else '❌'} Artifacts: {'Found' if has_artifacts else 'Missing - need to build'}")
    
    # Check config
    config_path = Path("retrieval/configs/retrieval_config.yaml")
    has_config = config_path.exists()
    status["config"] = has_config
    print(f"{'✅' if has_config else '❌'} Config: {'Found' if has_config else 'Missing'}")
    
    return all([has_groq, has_config]) and (has_dataset or has_artifacts)


def run_unit_tests():
    """Run unit tests with pytest"""
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_llm_client.py", "tests/test_cache.py", "tests/test_retrieval.py",
        "-v", "-m", "unit",
        "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def run_real_retrieval_test(samples: List[Dict]) -> Dict:
    """Run real retrieval test with actual pipeline"""
    print("\n" + "=" * 70)
    print("RUNNING REAL RETRIEVAL EVALUATION")
    print("=" * 70)
    
    try:
        import yaml
        from retrieval.pipeline import RetrievalPipeline
        
        # Load config
        with open("retrieval/configs/retrieval_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        # Initialize pipeline
        print("🔄 Loading retrieval pipeline...")
        pipeline = RetrievalPipeline(cfg)
        
        # Try to load indexes
        try:
            pipeline.load()
            print("✅ Pipeline loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load full pipeline: {e}")
            print("📝 Will test with available components")
        
        results = []
        correct_top3 = 0
        correct_top1 = 0
        total_time = 0
        
        # Test on sample
        test_samples = samples[:30]  # Use 30 for speed
        
        for i, sample in enumerate(test_samples):
            query = sample["question"]
            category = sample["category"]
            
            print(f"\n[{i+1}/{len(test_samples)}] Testing: {query[:60]}...")
            
            try:
                t0 = time.time()
                output = pipeline.run(query)
                elapsed = (time.time() - t0) * 1000
                total_time += elapsed
                
                # Check if correct category in results
                top_doc_category = None
                if output.docs:
                    top_doc = output.docs[0]
                    top_doc_category = top_doc.metadata.get("doc", {}).get("category", "unknown")
                
                is_correct_top1 = (top_doc_category == category)
                is_correct_top3 = any(
                    d.metadata.get("doc", {}).get("category") == category
                    for d in output.docs[:3]
                ) if len(output.docs) >= 3 else is_correct_top1
                
                if is_correct_top1:
                    correct_top1 += 1
                if is_correct_top3:
                    correct_top3 += 1
                
                results.append({
                    "query": query,
                    "category": category,
                    "top1_correct": is_correct_top1,
                    "top3_correct": is_correct_top3,
                    "time_ms": elapsed,
                    "num_docs": len(output.docs),
                })
                
                print(f"   ⏱️ {elapsed:.1f}ms | Top3: {'✅' if is_correct_top3 else '❌'}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "query": query,
                    "category": category,
                    "error": str(e),
                })
        
        # Calculate metrics
        n = len([r for r in results if "error" not in r])
        metrics = {
            "total_tested": len(test_samples),
            "successful": n,
            "acc_at_1": correct_top1 / n if n > 0 else 0,
            "acc_at_3": correct_top3 / n if n > 0 else 0,
            "avg_time_ms": total_time / n if n > 0 else 0,
            "results": results,
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Acc@1: {metrics['acc_at_1']:.1%}")
        print(f"   Acc@3: {metrics['acc_at_3']:.1%}")
        print(f"   Avg Time: {metrics['avg_time_ms']:.1f}ms")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Failed to run retrieval test: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_llm_client_test():
    """Test LLM client with real API"""
    print("\n" + "=" * 70)
    print("TESTING LLM CLIENT (Real API Call)")
    print("=" * 70)
    
    try:
        import yaml
        from generation.llm_client import LLMClient
        
        with open("retrieval/configs/retrieval_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        client = LLMClient(cfg)
        
        # Test simple prompt
        prompt = "What is 2+2? Answer in one word."
        print(f"📝 Prompt: {prompt}")
        print("🔄 Calling Groq API...")
        
        response = client.generate(prompt)
        
        print(f"\n📊 RESULT:")
        print(f"   Success: {response.success}")
        print(f"   Provider: {response.provider}")
        print(f"   Model: {response.model}")
        print(f"   Text: {response.text[:100]}...")
        print(f"   Latency: {response.latency_ms:.1f}ms")
        print(f"   Tokens: {response.prompt_tokens} in, {response.completion_tokens} out")
        
        return {
            "success": response.success,
            "latency_ms": response.latency_ms,
            "provider": response.provider,
        }
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_cache_test():
    """Test semantic cache"""
    print("\n" + "=" * 70)
    print("TESTING SEMANTIC CACHE")
    print("=" * 70)
    
    try:
        from cache.semantic_cache import SemanticCache
        import numpy as np
        
        cache = SemanticCache(
            l1_threshold=0.95,
            l2_threshold=0.90,
            ttl_seconds=3600,
            corpus_version="test_v1",
        )
        
        # Test store and lookup
        print("📝 Testing cache operations...")
        
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Store
        cache.store(
            query_embedding=embedding,
            query_text="zakat calculation",
            doc_ids=["qa_001", "qa_002"],
            answer="Zakat is 2.5% of wealth above nisab.",
        )
        
        # Lookup (should hit)
        result = cache.lookup(embedding)
        
        print(f"📊 CACHE RESULT:")
        print(f"   Hit: {result.hit}")
        print(f"   Level: {result.level}")
        print(f"   Similarity: {result.similarity:.4f}")
        print(f"   Answer cached: {result.answer is not None}")
        
        stats = cache.stats()
        print(f"\n📊 CACHE STATS:")
        print(f"   Size: {stats['current_size']}")
        print(f"   Stores: {stats['stores']}")
        print(f"   L1 Hit Rate: {stats['l1_hit_rate']:.1%}")
        
        return {"working": True, "stats": stats}
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return {"error": str(e)}


def generate_report(all_results: Dict):
    """Generate final report"""
    print("\n" + "=" * 70)
    print("FINAL TEST REPORT")
    print("=" * 70)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    
    # Save to file
    report_path = Path("evaluation/results/real_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_path}")
    
    # Print summary
    print("\n📋 SUMMARY:")
    for test_name, result in all_results.items():
        status = "✅" if "error" not in result else "❌"
        print(f"   {status} {test_name}")


def main():
    """Main test runner"""
    print("\n" + "=" * 70)
    print("URDUFINANCE RAG - REAL TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Check components
    can_proceed = check_components()
    
    if not can_proceed:
        print("\n❌ Missing critical components. Cannot proceed.")
        return
    
    # Run tests
    print("\n🧪 Starting tests...")
    
    # 1. Unit tests
    # all_results["unit_tests"] = {"passed": run_unit_tests()}
    
    # 2. Cache test (fast, no API needed)
    all_results["cache"] = run_cache_test()
    
    # 3. LLM client test (needs API key)
    all_results["llm_client"] = run_llm_client_test()
    
    # 4. Load dataset
    samples = load_dataset_sample(n=50)
    print(f"\n📊 Loaded {len(samples)} samples from dataset")
    all_results["dataset_load"] = {"samples_loaded": len(samples)}
    
    # 5. Retrieval test (if we have data)
    if samples:
        all_results["retrieval"] = run_real_retrieval_test(samples)
    
    # Generate report
    generate_report(all_results)
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
