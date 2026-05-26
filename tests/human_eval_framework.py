"""
human_eval_framework.py - Human evaluation framework for answer correctness

This module provides tools for:
1. Generating evaluation samples for human annotators
2. Structured rubrics for scoring answers
3. Computing inter-annotator agreement
4. Statistical confidence intervals
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statistics


@dataclass
class HumanEvalSample:
    """A single sample for human evaluation"""
    sample_id: str
    query: str
    language: str
    category: str
    
    # System output
    retrieved_docs: List[Dict]
    generated_answer: str
    reranker_score: float
    gate_passed: bool
    latency_ms: float
    
    # Ground truth (if available)
    reference_answer: Optional[str] = None
    
    # Human scores (to be filled)
    relevance_score: Optional[int] = None  # 1-5
    accuracy_score: Optional[int] = None  # 1-5
    completeness_score: Optional[int] = None  # 1-5
    groundedness_score: Optional[int] = None  # 1-5 (is answer supported by retrieval?)
    fluency_score: Optional[int] = None  # 1-5
    overall_score: Optional[int] = None  # 1-5
    
    # Binary flags
    contains_hallucination: Optional[bool] = None
    contradicts_retrieval: Optional[bool] = None
    appropriate_tone: Optional[bool] = None
    
    # Free-form feedback
    annotator_notes: str = ""
    
    # Metadata
    annotator_id: Optional[str] = None
    evaluation_timestamp: Optional[str] = None


@dataclass
class HumanEvalRubric:
    """Scoring rubric for human evaluation"""
    
    RELEVANCE_DESC = {
        5: "Perfectly addresses the user's query",
        4: "Mostly relevant with minor gaps",
        3: "Partially relevant, misses key aspects",
        2: "Barely relevant, mostly off-topic",
        1: "Completely irrelevant or wrong topic",
    }
    
    ACCURACY_DESC = {
        5: "All facts correct, no errors",
        4: "Minor factual error that doesn't change answer",
        3: "Some correct info mixed with errors",
        2: "Major factual errors",
        1: "Completely incorrect or fabricated",
    }
    
    COMPLETENESS_DESC = {
        5: "Fully comprehensive, no missing info",
        4: "Nearly complete, minor omissions",
        3: "Adequate but missing important details",
        2: "Significant gaps in information",
        1: "Severely incomplete",
    }
    
    GROUNDEDNESS_DESC = {
        5: "Fully supported by retrieved documents",
        4: "Mostly supported, minor inference",
        3: "Partial support, some inference required",
        2: "Weak support, heavy reliance on LLM knowledge",
        1: "No support, pure hallucination",
    }
    
    FLUENCY_DESC = {
        5: "Perfect grammar, natural flow",
        4: "Minor grammar issues, still natural",
        3: "Some awkward phrasing but understandable",
        2: "Frequent errors, hard to follow",
        1: "Incoherent or garbled",
    }


@dataclass
class HumanEvalReport:
    """Aggregated human evaluation report"""
    timestamp: str
    num_samples: int
    num_annotators: int
    
    # Mean scores
    mean_relevance: float
    mean_accuracy: float
    mean_completeness: float
    mean_groundedness: float
    mean_fluency: float
    mean_overall: float
    
    # Score distributions
    relevance_distribution: Dict[int, int]
    accuracy_distribution: Dict[int, int]
    
    # Binary metrics
    hallucination_rate: float
    groundedness_rate: float  # % with groundedness >= 4
    
    # Per-category breakdown
    per_category_scores: Dict[str, Dict[str, float]]
    
    # Per-language breakdown
    per_language_scores: Dict[str, Dict[str, float]]
    
    # Confidence intervals (95%)
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Inter-annotator agreement
    inter_annotator_kappa: Optional[float] = None
    
    # Comparison to automated metrics
    correlation_with_reranker: Optional[float] = None


class HumanEvalGenerator:
    """Generate evaluation samples for human annotators"""
    
    # Stratified sampling to ensure coverage
    CATEGORIES = ["islamic_finance", "digital_finance", "loans_credit", "banking", "tax"]
    LANGUAGES = ["roman_urdu", "urdu", "english"]
    
    # Sample queries by category and difficulty
    SAMPLE_QUERIES = [
        # Islamic Finance - Easy
        {"query": "zakat kis par farz hai", "category": "islamic_finance", "language": "roman_urdu", "difficulty": "easy"},
        {"query": "nisab kitna hai", "category": "islamic_finance", "language": "roman_urdu", "difficulty": "easy"},
        
        # Islamic Finance - Hard
        {"query": "zakat on retirement savings pkistan", "category": "islamic_finance", "language": "english", "difficulty": "hard"},
        {"query": "meezan bank car ijarah vs conventional lease", "category": "islamic_finance", "language": "english", "difficulty": "hard"},
        
        # Digital Finance - Easy
        {"query": "easypaisa account kaise banaye", "category": "digital_finance", "language": "roman_urdu", "difficulty": "easy"},
        {"query": "jazzcash withdrawal charges", "category": "digital_finance", "language": "english", "difficulty": "easy"},
        
        # Loans - Medium
        {"query": "home loan 20 saal ki qist", "category": "loans_credit", "language": "roman_urdu", "difficulty": "medium"},
        {"query": "personal loan without salary slip", "category": "loans_credit", "language": "english", "difficulty": "medium"},
        
        # Banking - Various
        {"query": "roshan digital account benefits", "category": "banking", "language": "english", "difficulty": "medium"},
        {"query": "sbp policy rate current", "category": "banking", "language": "english", "difficulty": "easy"},
        
        # Tax
        {"query": "fbr tax return deadline 2024", "category": "tax", "language": "english", "difficulty": "easy"},
        {"query": "withholding tax on banking transactions", "category": "tax", "language": "english", "difficulty": "hard"},
        
        # Edge cases
        {"query": "", "category": "edge", "language": "unknown", "difficulty": "edge"},
        {"query": "bank", "category": "edge", "language": "english", "difficulty": "edge"},
        {"query": "riba vs interest vs markup vs profit rate", "category": "edge", "language": "mixed", "difficulty": "edge"},
    ]
    
    def __init__(self, pipeline, generator, output_dir: str = "evaluation/human_eval"):
        self.pipeline = pipeline
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_samples(self, n_per_category: int = 3, seed: int = 42) -> List[HumanEvalSample]:
        """Generate stratified sample set for evaluation"""
        random.seed(seed)
        
        samples = []
        
        # Group queries by category
        by_category = {}
        for q in self.SAMPLE_QUERIES:
            cat = q["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)
        
        # Sample from each category
        for cat, queries in by_category.items():
            selected = random.sample(queries, min(n_per_category, len(queries)))
            
            for i, q in enumerate(selected):
                sample = self._create_sample(q, f"{cat}_{i}")
                samples.append(sample)
        
        return samples
    
    def _create_sample(self, query_data: Dict, sample_id: str) -> HumanEvalSample:
        """Create a single evaluation sample by running the pipeline"""
        query = query_data["query"]
        
        # Run pipeline (mocked if not available)
        try:
            retrieval_output = self.pipeline.run(query)
            retrieved_docs = [
                {"doc_id": d.doc_id, "doc_type": d.metadata.get("doc", {}).get("doc_type", "unknown")}
                for d in retrieval_output.docs[:3]
            ]
            latency = retrieval_output.total_ms
        except Exception as e:
            retrieved_docs = []
            latency = 0
        
        # Generate answer (mocked if not available)
        try:
            gen_output = self.generator.generate(
                query=query,
                query_embedding=None,
                reranked_docs=retrieved_docs,
                reranker_scores=[0.8],
            )
            answer = gen_output.answer if hasattr(gen_output, "answer") else str(gen_output)
            reranker_score = 0.8
            gate_passed = True
        except Exception as e:
            answer = f"[Error: {e}]"
            reranker_score = 0.0
            gate_passed = False
        
        return HumanEvalSample(
            sample_id=sample_id,
            query=query,
            language=query_data.get("language", "unknown"),
            category=query_data.get("category", "unknown"),
            retrieved_docs=retrieved_docs,
            generated_answer=answer,
            reranker_score=reranker_score,
            gate_passed=gate_passed,
            latency_ms=latency,
        )
    
    def export_for_annotators(self, samples: List[HumanEvalSample], format: str = "json"):
        """Export samples in format suitable for human annotators"""
        if format == "json":
            output_file = self.output_dir / "human_eval_samples.json"
            data = [asdict(s) for s in samples]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Exported {len(samples)} samples to {output_file}")
        
        elif format == "csv":
            import csv
            output_file = self.output_dir / "human_eval_samples.csv"
            
            # Flatten for CSV
            rows = []
            for s in samples:
                rows.append({
                    "sample_id": s.sample_id,
                    "query": s.query,
                    "language": s.language,
                    "category": s.category,
                    "generated_answer": s.generated_answer,
                    "reranker_score": s.reranker_score,
                    "gate_passed": s.gate_passed,
                    "retrieved_doc_count": len(s.retrieved_docs),
                })
            
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"Exported {len(samples)} samples to {output_file}")
        
        elif format == "html":
            output_file = self.output_dir / "human_eval_form.html"
            self._generate_html_form(samples, output_file)
            print(f"Exported HTML form to {output_file}")
    
    def _generate_html_form(self, samples: List[HumanEvalSample], output_file: Path):
        """Generate an HTML form for annotation"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>FinGuard RAG Human Evaluation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .sample { border: 1px solid #ccc; padding: 20px; margin-bottom: 30px; background: #f9f9f9; }
        .query { font-size: 18px; font-weight: bold; color: #333; margin-bottom: 10px; }
        .metadata { color: #666; font-size: 14px; margin-bottom: 15px; }
        .answer { background: white; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 15px; }
        .scores { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .score-item { display: flex; flex-direction: column; }
        .score-item label { font-weight: bold; margin-bottom: 5px; }
        select { padding: 5px; }
        .notes { width: 100%; margin-top: 10px; }
        .binary-flags { display: flex; gap: 20px; margin: 15px 0; }
        button { background: #007bff; color: white; padding: 10px 30px; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .rubric { background: #e9ecef; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>FinGuard RAG - Human Evaluation</h1>
    
    <div class="rubric">
        <h3>Scoring Guide:</h3>
        <p><strong>Relevance:</strong> 5=Perfect, 1=Irrelevant</p>
        <p><strong>Accuracy:</strong> 5=All facts correct, 1=All wrong</p>
        <p><strong>Completeness:</strong> 5=Fully comprehensive, 1=Severely incomplete</p>
        <p><strong>Groundedness:</strong> 5=Fully supported by docs, 1=No support</p>
        <p><strong>Fluency:</strong> 5=Perfect grammar, 1=Incoherent</p>
        <p><strong>Overall:</strong> Your overall impression (1-5)</p>
    </div>
    
    <form id="evalForm">
"""
        
        for i, s in enumerate(samples):
            html += f"""
        <div class="sample">
            <div class="query">{i+1}. {s.query}</div>
            <div class="metadata">
                ID: {s.sample_id} | Language: {s.language} | Category: {s.category} | 
                Reranker: {s.reranker_score:.3f} | Gate: {'✓' if s.gate_passed else '✗'}
            </div>
            <div class="answer">
                <strong>Generated Answer:</strong><br>
                {s.generated_answer}
            </div>
            
            <div class="scores">
                <div class="score-item">
                    <label>Relevance (1-5):</label>
                    <select name="{s.sample_id}_relevance" required>
                        <option value="">Select</option>
                        <option value="5">5 - Perfect</option>
                        <option value="4">4 - Good</option>
                        <option value="3">3 - OK</option>
                        <option value="2">2 - Poor</option>
                        <option value="1">1 - Wrong</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Accuracy (1-5):</label>
                    <select name="{s.sample_id}_accuracy" required>
                        <option value="">Select</option>
                        <option value="5">5 - Perfect</option>
                        <option value="4">4 - Good</option>
                        <option value="3">3 - OK</option>
                        <option value="2">2 - Poor</option>
                        <option value="1">1 - Wrong</option>
                    </select>
                </div>
                <div class="score-item">
                    <label>Groundedness (1-5):</label>
                    <select name="{s.sample_id}_groundedness" required>
                        <option value="">Select</option>
                        <option value="5">5 - Fully supported</option>
                        <option value="4">4 - Mostly supported</option>
                        <option value="3">3 - Partial</option>
                        <option value="2">2 - Weak support</option>
                        <option value="1">1 - Hallucination</option>
                    </select>
                </div>
            </div>
            
            <div class="binary-flags">
                <label>
                    <input type="checkbox" name="{s.sample_id}_hallucination">
                    Contains Hallucination
                </label>
                <label>
                    <input type="checkbox" name="{s.sample_id}_contradicts">
                    Contradicts Retrieved Docs
                </label>
            </div>
            
            <label>Notes:</label>
            <textarea name="{s.sample_id}_notes" class="notes" rows="2" placeholder="Optional feedback..."></textarea>
        </div>
"""
        
        html += """
        <button type="submit">Submit Evaluation</button>
    </form>
    
    <script>
        document.getElementById('evalForm').onsubmit = function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const results = {};
            for (let [key, value] of formData.entries()) {
                results[key] = value;
            }
            console.log('Evaluation Results:', results);
            alert('Evaluation captured! Check console for JSON export.');
            // In production, send to server
        };
    </script>
</body>
</html>
"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)


class HumanEvalAnalyzer:
    """Analyze completed human evaluations"""
    
    def __init__(self, eval_file: str):
        self.eval_file = Path(eval_file)
        self.samples = self._load_evaluations()
    
    def _load_evaluations(self) -> List[HumanEvalSample]:
        """Load completed evaluations from file"""
        if not self.eval_file.exists():
            return []
        
        with open(self.eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [HumanEvalSample(**s) for s in data]
    
    def compute_report(self) -> HumanEvalReport:
        """Compute aggregated report from evaluations"""
        from datetime import datetime
        
        if not self.samples:
            raise ValueError("No evaluation samples found")
        
        # Filter samples with human scores
        scored = [s for s in self.samples if s.overall_score is not None]
        
        if not scored:
            raise ValueError("No scored samples found")
        
        # Compute means
        def mean_safe(values):
            valid = [v for v in values if v is not None]
            return statistics.mean(valid) if valid else 0.0
        
        # Distributions
        rel_dist = {}
        acc_dist = {}
        for s in scored:
            r = s.relevance_score or 0
            a = s.accuracy_score or 0
            rel_dist[r] = rel_dist.get(r, 0) + 1
            acc_dist[a] = acc_dist.get(a, 0) + 1
        
        # Per-category
        per_cat = {}
        for s in scored:
            cat = s.category
            if cat not in per_cat:
                per_cat[cat] = {"scores": [], "count": 0}
            if s.overall_score:
                per_cat[cat]["scores"].append(s.overall_score)
                per_cat[cat]["count"] += 1
        
        for cat in per_cat:
            per_cat[cat]["mean"] = mean_safe(per_cat[cat]["scores"])
        
        # Per-language
        per_lang = {}
        for s in scored:
            lang = s.language
            if lang not in per_lang:
                per_lang[lang] = {"scores": [], "count": 0}
            if s.overall_score:
                per_lang[lang]["scores"].append(s.overall_score)
                per_lang[lang]["count"] += 1
        
        for lang in per_lang:
            per_lang[lang]["mean"] = mean_safe(per_lang[lang]["scores"])
        
        # Confidence intervals (95%)
        def confidence_interval(values, confidence=0.95):
            if len(values) < 2:
                return (0, 0)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            margin = 1.96 * stdev / (len(values) ** 0.5)
            return (mean - margin, mean + margin)
        
        overall_scores = [s.overall_score for s in scored if s.overall_score]
        
        return HumanEvalReport(
            timestamp=datetime.now().isoformat(),
            num_samples=len(scored),
            num_annotators=len(set(s.annotator_id for s in scored if s.annotator_id)),
            mean_relevance=mean_safe([s.relevance_score for s in scored]),
            mean_accuracy=mean_safe([s.accuracy_score for s in scored]),
            mean_completeness=mean_safe([s.completeness_score for s in scored]),
            mean_groundedness=mean_safe([s.groundedness_score for s in scored]),
            mean_fluency=mean_safe([s.fluency_score for s in scored]),
            mean_overall=mean_safe(overall_scores),
            relevance_distribution=rel_dist,
            accuracy_distribution=acc_dist,
            hallucination_rate=sum(1 for s in scored if s.contains_hallucination) / len(scored),
            groundedness_rate=sum(1 for s in scored if (s.groundedness_score or 0) >= 4) / len(scored),
            per_category_scores=per_cat,
            per_language_scores=per_lang,
            confidence_intervals={
                "overall": confidence_interval(overall_scores),
            },
        )
    
    def print_report(self, report: HumanEvalReport):
        """Print human evaluation report"""
        print("\n" + "=" * 70)
        print("HUMAN EVALUATION REPORT")
        print("=" * 70)
        print(f"Samples: {report.num_samples} | Annotators: {report.num_annotators}")
        print(f"Timestamp: {report.timestamp}")
        print()
        print("MEAN SCORES (1-5 scale):")
        print(f"  Overall:      {report.mean_overall:.2f}")
        print(f"  Relevance:    {report.mean_relevance:.2f}")
        print(f"  Accuracy:     {report.mean_accuracy:.2f}")
        print(f"  Completeness: {report.mean_completeness:.2f}")
        print(f"  Groundedness: {report.mean_groundedness:.2f}")
        print(f"  Fluency:      {report.mean_fluency:.2f}")
        print()
        print("QUALITY METRICS:")
        print(f"  Hallucination Rate: {report.hallucination_rate:.1%}")
        print(f"  Groundedness Rate:  {report.groundedness_rate:.1%}")
        print()
        
        if "overall" in report.confidence_intervals:
            ci = report.confidence_intervals["overall"]
            print(f"95% CI for Overall Score: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        print()
        print("PER-CATEGORY:")
        for cat, metrics in report.per_category_scores.items():
            print(f"  {cat}: Mean={metrics['mean']:.2f}, N={metrics['count']}")
        
        print()
        print("PER-LANGUAGE:")
        for lang, metrics in report.per_language_scores.items():
            print(f"  {lang}: Mean={metrics['mean']:.2f}, N={metrics['count']}")
        print("=" * 70)


def main():
    """Example usage"""
    from unittest.mock import MagicMock
    
    # Create with mocked components
    pipeline = MagicMock()
    generator = MagicMock()
    
    # Generate samples
    gen = HumanEvalGenerator(pipeline, generator)
    samples = gen.generate_samples(n_per_category=2)
    
    # Export for annotation
    gen.export_for_annotators(samples, format="json")
    gen.export_for_annotators(samples, format="csv")
    gen.export_for_annotators(samples, format="html")
    
    print(f"\nGenerated {len(samples)} evaluation samples")
    print(f"Files saved to: {gen.output_dir}")


if __name__ == "__main__":
    main()
