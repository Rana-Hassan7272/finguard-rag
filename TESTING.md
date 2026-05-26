# FinGuard RAG - Testing & Evaluation

This document describes the comprehensive testing suite for FinGuard RAG, including unit tests, adversarial evaluation, and human evaluation frameworks.

## Test Suite Overview

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_llm_client.py       # LLM client tests with mocked APIs
├── test_cache.py            # Semantic cache tests
├── test_retrieval.py        # Retrieval pipeline tests
├── test_adversarial.py      # Adversarial test wrappers
├── adversarial_eval.py      # Full adversarial evaluation suite
├── human_eval_framework.py  # Human evaluation framework
└── __init__test_runner.py   # Standalone test runner
```

## Running Tests

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all unit tests (fast)
python -m pytest tests/ -v -m "unit"

# Run with coverage (aims for 80%+)
python -m pytest tests/ --cov=finguard --cov-report=html --cov-fail-under=80

# Run adversarial evaluation
python -m tests.adversarial_eval

# Run human evaluation sample generation
python -m tests.human_eval_framework
```

### Test Categories

| Category | Marker | Description | Speed |
|----------|--------|-------------|-------|
| Unit | `unit` | Fast, isolated, mocked | < 10s |
| Integration | `integration` | May call real APIs | 30s-2min |
| Slow | `slow` | Large-scale tests | > 1min |
| Adversarial | `adversarial` | Robustness testing | ~1min |
| Cache | `cache` | Cache-specific tests | < 5s |
| Retrieval | `retrieval` | Pipeline tests | < 10s |

## Test Coverage

Current coverage targets:

- **Overall**: 80% minimum
- **Core modules** (cache, llm_client): 90%+
- **Retrieval**: 75%+

View coverage report:
```bash
pytest tests/ --cov=finguard --cov-report=html
# Open htmlcov/index.html
```

## Adversarial Evaluation

The adversarial evaluation suite (`adversarial_eval.py`) provides realistic robustness testing:

### Test Types

1. **Typo Resilience**: Queries with realistic Roman Urdu/English typos
2. **Paraphrase Testing**: Same intent, different wording
3. **Edge Cases**: Empty queries, very long queries, mixed languages
4. **Out-of-Scope**: Non-financial queries (should be rejected)
5. **Negated Queries**: Tricky phrasings that confuse simple classifiers

### Running Adversarial Tests

```bash
# Full adversarial evaluation
python -m tests.adversarial_eval

# Output saved to: evaluation/results/adversarial_report.json
```

### Expected Results (Realistic)

| Metric | Range | Notes |
|--------|-------|-------|
| Acc@1 | 65-80% | Top document relevant |
| Acc@3 | 75-90% | Correct info in top 3 |
| MRR | 0.75-0.85 | Mean reciprocal rank |
| Typos | 60-75% | With spelling errors |
| Paraphrases | 70-85% | Different wording |
| Out-of-scope | >90% | Correctly rejected |

## Human Evaluation Framework

For realistic answer quality assessment:

```bash
# Generate samples for annotation
python -m tests.human_eval_framework

# Creates:
# - evaluation/human_eval/human_eval_samples.json
# - evaluation/human_eval/human_eval_samples.csv
# - evaluation/human_eval/human_eval_form.html
```

### Scoring Rubric (1-5 scale)

| Dimension | 5 (Best) | 1 (Worst) |
|-----------|----------|-----------|
| Relevance | Perfectly addresses query | Completely irrelevant |
| Accuracy | All facts correct | All facts wrong |
| Completeness | Fully comprehensive | Severely incomplete |
| Groundedness | Fully supported by docs | Pure hallucination |
| Fluency | Perfect grammar | Incoherent |

### Realistic Human Evaluation Targets

| Metric | Target | Industry Typical |
|--------|--------|------------------|
| Mean Overall | >3.5 | 3.0-4.0 |
| Hallucination Rate | <15% | 10-25% |
| Groundedness Rate | >75% | 60-80% |

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/tests.yml`):

- Runs on Python 3.10, 3.11
- Generates coverage reports
- Creates coverage badge
- Runs linting checks

## Key Design Decisions

### Why Mocked LLM Tests?

Real LLM APIs are:
- Expensive at scale
- Non-deterministic (hard to assert)
- Slow (hurt dev velocity)

We mock API calls but test all retry/fallback logic.

### Why Adversarial Testing?

The original 100% accuracy claim was on a small, clean test set. Real-world queries include:
- Typos (Roman Urdu has no standard spelling)
- Paraphrases
- Ambiguous intent
- Out-of-domain

Adversarial testing provides realistic robustness metrics.

### Why Human Evaluation?

Automated metrics (like reranker scores) don't capture:
- Factual correctness
- Answer completeness
- Tone appropriateness
- Cultural/language nuance

Human eval provides ground truth for answer quality.

## Known Limitations

1. **Test Data**: Some tests use mocked data rather than full corpus
2. **Integration**: Real API tests require valid API keys
3. **Coverage**: UI/app code has lower coverage (integration-tested via HF Space)

## Contributing

When adding features:
1. Add unit tests with mocks
2. Add adversarial test cases if changing retrieval/generation
3. Ensure coverage doesn't drop below 80%
4. Update this doc if changing testing approach
