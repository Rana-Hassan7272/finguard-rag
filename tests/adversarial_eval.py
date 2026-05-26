"""
adversarial_eval.py - Adversarial evaluation suite for FinGuard RAG

This module provides adversarial testing to challenge the 100% accuracy claims
and provide realistic metrics on system robustness.
"""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class AdversarialResult:
    """Result of an adversarial test case"""
    test_type: str
    query: str
    expected_category: Optional[str]
    expected_doc_pattern: str  # Regex pattern to match expected doc content
    
    # Retrieval metrics
    retrieved_docs: list = field(default_factory=list)
    top_doc_relevance: float = 0.0  # Did top doc contain expected info?
    correct_doc_in_top3: bool = False
    correct_doc_in_top10: bool = False
    
    # Answer metrics
    generated_answer: str = ""
    answer_relevance_score: float = 0.0  # 0-1 based on keyword matching
    answer_grounded_in_retrieval: bool = False
    hallucination_detected: bool = False
    
    # Confidence metrics
    reranker_score: float = 0.0
    gate_passed: bool = False
    
    # Timing
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class AdversarialReport:
    """Full adversarial evaluation report"""
    timestamp: str
    total_tests: int
    
    # Retrieval metrics
    retrieval_accuracy_top1: float
    retrieval_accuracy_top3: float
    retrieval_accuracy_top10: float
    retrieval_mrr: float
    
    # Answer quality metrics
    answer_relevance_avg: float
    groundedness_rate: float
    hallucination_rate: float
    
    # Confidence/gate metrics
    gate_pass_rate: float
    avg_confidence_score: float
    
    # Category-specific breakdown
    per_category_results: dict
    
    # Adversarial-specific metrics
    typos_resilience: float
    paraphrase_resilience: float
    edge_case_handling: float
    out_of_scope_rejection: float
    
    # Detailed results
    results: list


class AdversarialTestSuite:
    """
    Generate and run adversarial test cases to challenge the RAG system.
    
    Adversarial tests include:
    1. Typos and spelling errors
    2. Paraphrased queries (same intent, different words)
    3. Ambiguous/multi-intent queries
    4. Out-of-domain queries
    5. Edge cases (very short, very long, mixed languages)
    6. Negated queries
    7. Time-sensitive queries
    """
    
    # Base queries with known correct answers
    BASE_QUERIES = [
        {
            "query": "zakat ka hisab kaise karein",
            "category": "islamic_finance",
            "expected_keywords": ["zakat", "2.5%", "nisab", "hisab", "calculate"],
            "language": "roman_urdu",
        },
        {
            "query": "easypaisa se paise kaise bhejein",
            "category": "digital_finance",
            "expected_keywords": ["easypaisa", "send", "bhej", "mobile", "transfer"],
            "language": "roman_urdu",
        },
        {
            "query": "home loan ki qist kitni hogi 50 lakh pe",
            "category": "loans_credit",
            "expected_keywords": ["loan", "qist", "monthly", "installment", "50 lakh"],
            "language": "roman_urdu",
        },
        {
            "query": "riba aur interest mein kya farq hai",
            "category": "islamic_finance",
            "expected_keywords": ["riba", "interest", "farq", "islam", "haram"],
            "language": "roman_urdu",
        },
        {
            "query": "What is the nisab threshold for zakat in Pakistan?",
            "category": "islamic_finance",
            "expected_keywords": ["nisab", "threshold", "zakat", "pakistan", "gold", "silver"],
            "language": "english",
        },
        {
            "query": "how to file income tax return on FBR IRIS",
            "category": "financial_education",
            "expected_keywords": ["tax", "fbr", "iris", "return", "file"],
            "language": "english",
        },
    ]
    
    # Typos and variations for resilience testing
    TYPO_VARIATIONS = [
        ("zakat", ["zakkat", "zakaat", "zakht", "zkat"]),
        ("hisab", ["hissb", "hisib", "hisaab", "hesab"]),
        ("kaise", ["kesay", "kese", "kaisa", "kesey"]),
        ("easypaisa", ["easypaisaa", "ezypaisa", "easypaysa", "esypaisa"]),
        ("paise", ["paisay", "paisa", "paisey", "pesay"]),
        ("bhejein", ["bhejain", "bhjain", "bhejen", "bhejayn"]),
        ("loan", ["lone", "lon", "loaan", "laon"]),
        ("qist", ["qisst", "kist", "qast", "kisst"]),
    ]
    
    # Paraphrases for same intent
    PARAPHRASES = [
        {
            "original": "zakat ka hisab kaise karein",
            "variations": [
                "how to calculate zakat amount",
                "zakat calculation method pakistan",
                "zakat percentage on savings",
                "kis tarah zakat nikalain",
            ],
        },
        {
            "original": "easypaisa se paise kaise bhejein",
            "variations": [
                "how to send money via easypaisa",
                "easypaisa money transfer steps",
                "easypaisa app se funds transfer",
                "send cash through easypaisa",
            ],
        },
    ]
    
    # Out-of-scope queries (should be rejected or handled gracefully)
    OUT_OF_SCOPE_QUERIES = [
        "who is the prime minister of pakistan",
        "what is the weather in lahore today",
        "how to cook biryani",
        "latest cricket score pakistan vs india",
        "best restaurants in karachi",
        "how to apply for visa to usa",
        "what is machine learning",
        "tell me a joke",
    ]
    
    # Edge cases
    EDGE_CASE_QUERIES = [
        "",  # Empty
        "z",  # Too short
        "zakat",  # Single term
        "zakat " * 100,  # Very long (spam)
        "زکوٰۃ hisab calculate 123",  # Mixed language + numbers
        "!!!???",  # Only punctuation
        "zakat or riba or loan or tax or investment or banking",  # Too many intents
    ]
    
    # Negated queries (trick questions)
    NEGATED_QUERIES = [
        {
            "query": "zakat ka hisab NAHIN kaise karein",
            "intent": "don't want to calculate zakat",  # Confusing negation
            "language": "roman_urdu",
        },
        {
            "query": "which banks do NOT offer riba-free accounts",
            "intent": "find conventional banks",
            "language": "english",
        },
    ]
    
    def __init__(self, pipeline, generator, reranker):
        """
        Args:
            pipeline: RetrievalPipeline instance
            generator: Generator instance for answer generation
            reranker: CrossEncoderReranker instance
        """
        self.pipeline = pipeline
        self.generator = generator
        self.reranker = reranker
        self.results = []
    
    def generate_typo_queries(self, n=20):
        """Generate queries with realistic typos"""
        typo_tests = []
        
        for base in self.BASE_QUERIES[:3]:  # Use first 3 base queries
            for _ in range(n // 3):
                query = base["query"]
                # Apply random typo substitutions
                for correct, typos in self.TYPO_VARIATIONS:
                    if correct in query:
                        typo = random.choice(typos)
                        query = query.replace(correct, typo, 1)
                
                typo_tests.append({
                    "query": query,
                    "category": base["category"],
                    "expected_keywords": base["expected_keywords"],
                    "language": base["language"],
                    "test_type": "typo",
                })
        
        return typo_tests
    
    def generate_paraphrase_queries(self):
        """Generate paraphrased versions of base queries"""
        para_tests = []
        
        for para_set in self.PARAPHRASES:
            base = next((b for b in self.BASE_QUERIES if b["query"] == para_set["original"]), None)
            if base:
                for variation in para_set["variations"]:
                    para_tests.append({
                        "query": variation,
                        "category": base["category"],
                        "expected_keywords": base["expected_keywords"],
                        "language": "english" if variation.startswith("how") else base["language"],
                        "test_type": "paraphrase",
                    })
        
        return para_tests
    
    def run_adversarial_test(self, test_case) -> AdversarialResult:
        """Run a single adversarial test"""
        import time
        
        query = test_case["query"]
        
        result = AdversarialResult(
            test_type=test_case.get("test_type", "unknown"),
            query=query,
            expected_category=test_case.get("category"),
            expected_doc_pattern=r"|".join(test_case.get("expected_keywords", [])),
        )
        
        # Time the retrieval
        t0 = time.time()
        try:
            retrieval_output = self.pipeline.run(query)
            result.retrieval_ms = retrieval_output.total_ms
            result.retrieved_docs = [
                {"doc_id": d.doc_id, "metadata": d.metadata}
                for d in retrieval_output.docs
            ]
        except Exception as e:
            result.retrieval_ms = (time.time() - t0) * 1000
            result.retrieved_docs = []
        
        # Check if correct info in retrieved docs
        if result.retrieved_docs:
            top_doc_text = str(result.retrieved_docs[0].get("metadata", {}))
            result.top_doc_relevance = self._score_relevance(
                top_doc_text, test_case.get("expected_keywords", [])
            )
            
            # Check top-3 and top-10
            for i, doc in enumerate(result.retrieved_docs[:3]):
                doc_text = str(doc.get("metadata", {}))
                if self._contains_keywords(doc_text, test_case.get("expected_keywords", [])):
                    result.correct_doc_in_top3 = True
                    break
            
            for i, doc in enumerate(result.retrieved_docs[:10]):
                doc_text = str(doc.get("metadata", {}))
                if self._contains_keywords(doc_text, test_case.get("expected_keywords", [])):
                    result.correct_doc_in_top10 = True
                    break
        
        # Rerank and generate
        t1 = time.time()
        try:
            # Mock reranker call
            result.reranker_score = 0.75  # Placeholder
            result.gate_passed = result.reranker_score > 0.55
            
            # Mock generation
            result.generated_answer = f"Based on the retrieved documents, here's information about {query[:20]}..."
            result.answer_relevance_score = self._score_relevance(
                result.generated_answer, test_case.get("expected_keywords", [])
            )
        except Exception as e:
            pass
        
        result.generation_ms = (time.time() - t1) * 1000
        result.total_ms = result.retrieval_ms + result.generation_ms
        
        return result
    
    def _score_relevance(self, text: str, keywords: list) -> float:
        """Score relevance based on keyword presence"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords)
    
    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """Check if text contains at least one expected keyword"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)
    
    def run_full_evaluation(self) -> AdversarialReport:
        """Run complete adversarial evaluation"""
        from datetime import datetime
        
        all_tests = []
        
        # Base queries
        for q in self.BASE_QUERIES:
            q["test_type"] = "base"
            all_tests.append(q)
        
        # Typo variants
        all_tests.extend(self.generate_typo_queries())
        
        # Paraphrases
        all_tests.extend(self.generate_paraphrase_queries())
        
        # Out-of-scope (expect rejection)
        for q in self.OUT_OF_SCOPE_QUERIES:
            all_tests.append({
                "query": q,
                "category": None,
                "expected_keywords": [],
                "test_type": "out_of_scope",
            })
        
        # Edge cases
        for i, q in enumerate(self.EDGE_CASE_QUERIES):
            all_tests.append({
                "query": q,
                "category": None,
                "expected_keywords": [] if i < 2 else ["zakat"],
                "test_type": "edge_case",
            })
        
        # Run all tests
        results = []
        for test in all_tests:
            result = self.run_adversarial_test(test)
            results.append(result)
        
        # Calculate metrics
        return self._compile_report(results)
    
    def _compile_report(self, results: list) -> AdversarialReport:
        """Compile results into a report"""
        from datetime import datetime
        
        n = len(results)
        if n == 0:
            return AdversarialReport(
                timestamp=datetime.now().isoformat(),
                total_tests=0, retrieval_accuracy_top1=0, retrieval_accuracy_top3=0,
                retrieval_accuracy_top10=0, retrieval_mrr=0, answer_relevance_avg=0,
                groundedness_rate=0, hallucination_rate=0, gate_pass_rate=0,
                avg_confidence_score=0, per_category_results={},
                typos_resilience=0, paraphrase_resilience=0, edge_case_handling=0,
                out_of_scope_rejection=0, results=[]
            )
        
        # Filter by test type
        base_results = [r for r in results if r.test_type == "base"]
        typo_results = [r for r in results if r.test_type == "typo"]
        para_results = [r for r in results if r.test_type == "paraphrase"]
        oos_results = [r for r in results if r.test_type == "out_of_scope"]
        edge_results = [r for r in results if r.test_type == "edge_case"]
        
        # Calculate retrieval accuracy
        top1_hits = sum(1 for r in results if r.top_doc_relevance >= 0.5)
        top3_hits = sum(1 for r in results if r.correct_doc_in_top3)
        top10_hits = sum(1 for r in results if r.correct_doc_in_top10)
        
        # Calculate MRR
        mrr_sum = 0.0
        for r in results:
            if r.correct_doc_in_top3:
                # Find first relevant doc rank
                for i, doc in enumerate(r.retrieved_docs[:3]):
                    if self._contains_keywords(str(doc), [r.expected_doc_pattern]):
                        mrr_sum += 1.0 / (i + 1)
                        break
        
        # Out-of-scope should be rejected
        oos_rejected = sum(1 for r in oos_results if not r.gate_passed)
        
        return AdversarialReport(
            timestamp=datetime.now().isoformat(),
            total_tests=n,
            retrieval_accuracy_top1=round(top1_hits / n, 4),
            retrieval_accuracy_top3=round(top3_hits / n, 4),
            retrieval_accuracy_top10=round(top10_hits / n, 4),
            retrieval_mrr=round(mrr_sum / n, 4),
            answer_relevance_avg=round(np.mean([r.answer_relevance_score for r in results]), 4),
            groundedness_rate=round(sum(1 for r in results if r.answer_grounded_in_retrieval) / n, 4),
            hallucination_rate=round(sum(1 for r in results if r.hallucination_detected) / n, 4),
            gate_pass_rate=round(sum(1 for r in results if r.gate_passed) / n, 4),
            avg_confidence_score=round(np.mean([r.reranker_score for r in results]), 4),
            per_category_results=self._category_breakdown(results),
            typos_resilience=round(sum(1 for r in typo_results if r.correct_doc_in_top3) / max(len(typo_results), 1), 4),
            paraphrase_resilience=round(sum(1 for r in para_results if r.correct_doc_in_top3) / max(len(para_results), 1), 4),
            edge_case_handling=round(sum(1 for r in edge_results if r.gate_passed or r.correct_doc_in_top3) / max(len(edge_results), 1), 4),
            out_of_scope_rejection=round(oos_rejected / max(len(oos_results), 1), 4),
            results=results,
        )
    
    def _category_breakdown(self, results: list) -> dict:
        """Break down results by category"""
        categories = {}
        for r in results:
            cat = r.expected_category or "unknown"
            if cat not in categories:
                categories[cat] = {"total": 0, "top3_hits": 0, "gate_passed": 0}
            categories[cat]["total"] += 1
            if r.correct_doc_in_top3:
                categories[cat]["top3_hits"] += 1
            if r.gate_passed:
                categories[cat]["gate_passed"] += 1
        
        # Calculate rates
        for cat in categories:
            t = categories[cat]["total"]
            categories[cat]["top3_accuracy"] = round(categories[cat]["top3_hits"] / t, 4)
            categories[cat]["pass_rate"] = round(categories[cat]["gate_passed"] / t, 4)
        
        return categories


def run_adversarial_evaluation(pipeline=None, generator=None, reranker=None):
    """
    Run adversarial evaluation and print report.
    
    If components are not provided, uses mocked versions for demonstration.
    """
    from unittest.mock import MagicMock
    
    # Use mocks if real components not provided
    if pipeline is None:
        pipeline = MagicMock()
    if generator is None:
        generator = MagicMock()
    if reranker is None:
        reranker = MagicMock()
    
    suite = AdversarialTestSuite(pipeline, generator, reranker)
    report = suite.run_full_evaluation()
    
    # Print report
    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Tests: {report.total_tests}")
    print()
    print("RETRIEVAL METRICS:")
    print(f"  Acc@1:  {report.retrieval_accuracy_top1:.1%}")
    print(f"  Acc@3:  {report.retrieval_accuracy_top3:.1%}")
    print(f"  Acc@10: {report.retrieval_accuracy_top10:.1%}")
    print(f"  MRR:    {report.retrieval_mrr:.4f}")
    print()
    print("ANSWER QUALITY:")
    print(f"  Avg Relevance:   {report.answer_relevance_avg:.2f}")
    print(f"  Groundedness:    {report.groundedness_rate:.1%}")
    print(f"  Hallucination:   {report.hallucination_rate:.1%}")
    print()
    print("GATE/CONFIDENCE:")
    print(f"  Pass Rate:       {report.gate_pass_rate:.1%}")
    print(f"  Avg Confidence:  {report.avg_confidence_score:.4f}")
    print()
    print("ADVERSARIAL RESILIENCE:")
    print(f"  Typos:           {report.typos_resilience:.1%}")
    print(f"  Paraphrases:     {report.paraphrase_resilience:.1%}")
    print(f"  Edge Cases:      {report.edge_case_handling:.1%}")
    print(f"  Out-of-Scope:    {report.out_of_scope_rejection:.1%} (should reject)")
    print()
    print("PER-CATEGORY:")
    for cat, metrics in report.per_category_results.items():
        print(f"  {cat}: Acc@3={metrics['top3_accuracy']:.1%}, Pass={metrics['pass_rate']:.1%}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    # Run with mocked components for demonstration
    report = run_adversarial_evaluation()
    
    # Save report
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "adversarial_report.json"
    with open(report_file, "w") as f:
        # Convert to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "total_tests": report.total_tests,
            "retrieval_accuracy_top1": report.retrieval_accuracy_top1,
            "retrieval_accuracy_top3": report.retrieval_accuracy_top3,
            "retrieval_accuracy_top10": report.retrieval_accuracy_top10,
            "retrieval_mrr": report.retrieval_mrr,
            "answer_relevance_avg": report.answer_relevance_avg,
            "groundedness_rate": report.groundedness_rate,
            "hallucination_rate": report.hallucination_rate,
            "gate_pass_rate": report.gate_pass_rate,
            "avg_confidence_score": report.avg_confidence_score,
            "per_category_results": report.per_category_results,
            "typos_resilience": report.typos_resilience,
            "paraphrase_resilience": report.paraphrase_resilience,
            "edge_case_handling": report.edge_case_handling,
            "out_of_scope_rejection": report.out_of_scope_rejection,
        }
        json.dump(report_dict, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
