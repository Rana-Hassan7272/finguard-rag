"""
Test runner script for local testing without pytest CLI
"""

import sys
import subprocess


def run_tests():
    """Run the full test suite"""
    args = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=finguard",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=80",
        "-m", "not integration and not slow",
    ]
    
    print("Running tests...")
    print(f"Command: {' '.join(args)}")
    result = subprocess.run(args)
    return result.returncode


def run_unit_tests():
    """Run only unit tests (fast)"""
    args = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-m", "unit",
    ]
    
    result = subprocess.run(args)
    return result.returncode


def run_with_coverage():
    """Run tests and generate coverage report"""
    args = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=finguard",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-fail-under=80",
    ]
    
    result = subprocess.run(args)
    return result.returncode


def run_adversarial():
    """Run adversarial evaluation"""
    result = subprocess.run([
        sys.executable,
        "-m",
        "tests.adversarial_eval",
    ])
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UrduFinance Test Runner")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "coverage", "adversarial"],
        default="all",
        help="Type of tests to run",
    )
    
    args = parser.parse_args()
    
    if args.type == "unit":
        sys.exit(run_unit_tests())
    elif args.type == "coverage":
        sys.exit(run_with_coverage())
    elif args.type == "adversarial":
        sys.exit(run_adversarial())
    else:
        sys.exit(run_tests())
