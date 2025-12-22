"""
Script to run RAG system evaluation: baseline and improved versions.

This script:
1. Runs baseline evaluation (no improvements)
2. Runs evaluation with Query Expansion
3. Runs evaluation with Re-ranking
4. Runs evaluation with Hybrid Search
5. Runs evaluation with both improvements
6. Compares results and generates comparison report
"""

import json
from pathlib import Path
from rag_evaluator import RAGEvaluator


def load_evaluation_results(file_path: Path) -> dict:
    """Load evaluation results from JSON file."""
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(baseline: dict, improved: dict, improvement_name: str) -> dict:
    """Compare baseline and improved results."""
    if not baseline or not improved:
        return None
    
    baseline_metrics = baseline.get('metrics', {})
    improved_metrics = improved.get('metrics', {})
    
    baseline_precision = baseline_metrics.get('average_precision_at_k', 0.0)
    improved_precision = improved_metrics.get('average_precision_at_k', 0.0)
    
    baseline_faithfulness = baseline_metrics.get('average_faithfulness_score', 0.0)
    improved_faithfulness = improved_metrics.get('average_faithfulness_score', 0.0)
    
    # Weak questions metrics (precision < 0.8)
    baseline_weak_precision = baseline_metrics.get('average_precision_weak_questions', 0.0)
    improved_weak_precision = improved_metrics.get('average_precision_weak_questions', 0.0)
    baseline_weak_count = baseline_metrics.get('weak_questions_count', 0)
    improved_weak_count = improved_metrics.get('weak_questions_count', 0)
    
    precision_improvement = ((improved_precision - baseline_precision) / baseline_precision * 100) if baseline_precision > 0 else 0.0
    faithfulness_improvement = ((improved_faithfulness - baseline_faithfulness) / baseline_faithfulness * 100) if baseline_faithfulness > 0 else 0.0
    weak_precision_improvement = ((improved_weak_precision - baseline_weak_precision) / baseline_weak_precision * 100) if baseline_weak_precision > 0 else 0.0
    
    return {
        "improvement_name": improvement_name,
        "baseline": {
            "precision_at_k": baseline_precision,
            "faithfulness_score": baseline_faithfulness,
            "weak_precision": baseline_weak_precision,
            "weak_count": baseline_weak_count
        },
        "improved": {
            "precision_at_k": improved_precision,
            "faithfulness_score": improved_faithfulness,
            "weak_precision": improved_weak_precision,
            "weak_count": improved_weak_count
        },
        "improvements": {
            "precision_improvement_percent": round(precision_improvement, 2),
            "faithfulness_improvement_percent": round(faithfulness_improvement, 2),
            "precision_absolute_improvement": round(improved_precision - baseline_precision, 4),
            "faithfulness_absolute_improvement": round(improved_faithfulness - baseline_faithfulness, 4),
            "weak_precision_improvement_percent": round(weak_precision_improvement, 2),
            "weak_precision_absolute_improvement": round(improved_weak_precision - baseline_weak_precision, 4),
            "weak_count_change": improved_weak_count - baseline_weak_count
        }
    }


def print_comparison(comparison: dict):
    """Print comparison results in a readable format."""
    if not comparison:
        return
    
    print(f"\n{'='*60}")
    print(f"Comparison: {comparison['improvement_name']}")
    print(f"{'='*60}")
    print(f"Retrieval Precision@K:")
    print(f"  Baseline:  {comparison['baseline']['precision_at_k']:.4f}")
    print(f"  Improved: {comparison['improved']['precision_at_k']:.4f}")
    print(f"  Improvement: {comparison['improvements']['precision_improvement_percent']:+.2f}% "
          f"({comparison['improvements']['precision_absolute_improvement']:+.4f})")
    print()
    print(f"Answer Faithfulness Score:")
    print(f"  Baseline:  {comparison['baseline']['faithfulness_score']:.4f}")
    print(f"  Improved: {comparison['improved']['faithfulness_score']:.4f}")
    print(f"  Improvement: {comparison['improvements']['faithfulness_improvement_percent']:+.2f}% "
          f"({comparison['improvements']['faithfulness_absolute_improvement']:+.4f})")
    print()
    if 'weak_precision' in comparison['baseline']:
        print(f"Weak Questions Precision (P<0.8):")
        print(f"  Baseline:  {comparison['baseline']['weak_precision']:.4f} ({comparison['baseline']['weak_count']} questions)")
        print(f"  Improved: {comparison['improved']['weak_precision']:.4f} ({comparison['improved']['weak_count']} questions)")
        print(f"  Improvement: {comparison['improvements']['weak_precision_improvement_percent']:+.2f}% "
              f"({comparison['improvements']['weak_precision_absolute_improvement']:+.4f})")
        print(f"  Weak Questions Count Change: {comparison['improvements']['weak_count_change']:+d}")
    print(f"{'='*60}\n")


def main():
    """Run full evaluation suite."""
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("RAG System Evaluation Suite")
    print("="*60)
    print("\nThis will run evaluations for:")
    print("1. Baseline (no improvements)")
    print("2. Query Expansion only")
    print("3. Re-ranking only")
    print("4. Hybrid Search only")
    print("5. Both improvements combined")
    print("\nNote: Make sure the RAG system is set up and Weaviate is running!")
    print("="*60)
    
    input("\nPress Enter to continue...")
    
    results = {}
    
    # 1. Baseline evaluation
    print("\n" + "="*60)
    print("1. Running BASELINE evaluation (no improvements)")
    print("="*60)
    evaluator_baseline = RAGEvaluator(use_query_expansion=False, use_reranking=False)
    baseline_results = evaluator_baseline.run_full_evaluation(
        output_file=script_dir / "evaluation_results_baseline.json"
    )
    results['baseline'] = baseline_results
    
    # 2. Query Expansion only
    print("\n" + "="*60)
    print("2. Running evaluation with QUERY EXPANSION")
    print("="*60)
    evaluator_qe = RAGEvaluator(use_query_expansion=True, use_reranking=False)
    qe_results = evaluator_qe.run_full_evaluation(
        output_file=script_dir / "evaluation_results_qe.json"
    )
    results['query_expansion'] = qe_results
    
    # 3. Re-ranking only
    print("\n" + "="*60)
    print("3. Running evaluation with RE-RANKING")
    print("="*60)
    evaluator_rr = RAGEvaluator(use_query_expansion=False, use_reranking=True)
    rr_results = evaluator_rr.run_full_evaluation(
        output_file=script_dir / "evaluation_results_rr.json"
    )
    results['reranking'] = rr_results
    
    # 4. Hybrid Search only
    print("\n" + "="*60)
    print("4. Running evaluation with HYBRID SEARCH")
    print("="*60)
    evaluator_hybrid = RAGEvaluator(use_query_expansion=False, use_reranking=False, use_hybrid_search=True)
    hybrid_results = evaluator_hybrid.run_full_evaluation(
        output_file=script_dir / "evaluation_results_hybrid.json"
    )
    results['hybrid'] = hybrid_results
    
    # 5. Both improvements
    print("\n" + "="*60)
    print("5. Running evaluation with BOTH IMPROVEMENTS (QE + RR)")
    print("="*60)
    evaluator_both = RAGEvaluator(use_query_expansion=True, use_reranking=True, use_hybrid_search=False)
    both_results = evaluator_both.run_full_evaluation(
        output_file=script_dir / "evaluation_results_qe_rr.json"
    )
    results['both'] = both_results
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    comparisons = {}
    
    # Compare Query Expansion vs Baseline
    comparison_qe = compare_results(
        baseline_results,
        qe_results,
        "Query Expansion vs Baseline"
    )
    if comparison_qe:
        comparisons['query_expansion'] = comparison_qe
        print_comparison(comparison_qe)
    
    # Compare Re-ranking vs Baseline
    comparison_rr = compare_results(
        baseline_results,
        rr_results,
        "Re-ranking vs Baseline"
    )
    if comparison_rr:
        comparisons['reranking'] = comparison_rr
        print_comparison(comparison_rr)
    
    # Compare Hybrid Search vs Baseline
    comparison_hybrid = compare_results(
        baseline_results,
        hybrid_results,
        "Hybrid Search vs Baseline"
    )
    if comparison_hybrid:
        comparisons['hybrid'] = comparison_hybrid
        print_comparison(comparison_hybrid)
    
    # Compare Both vs Baseline
    comparison_both = compare_results(
        baseline_results,
        both_results,
        "Both Improvements (QE+RR) vs Baseline"
    )
    if comparison_both:
        comparisons['both'] = comparison_both
        print_comparison(comparison_both)
    
    # Save comparison results
    comparison_file = script_dir / "evaluation_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            "comparisons": comparisons,
            "all_results": {
                "baseline": baseline_results.get('metrics', {}),
                "query_expansion": qe_results.get('metrics', {}),
                "reranking": rr_results.get('metrics', {}),
                "hybrid": hybrid_results.get('metrics', {}),
                "both": both_results.get('metrics', {})
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Comparison results saved to: {comparison_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline Precision@K: {baseline_results.get('metrics', {}).get('average_precision_at_k', 0.0):.4f}")
    baseline_weak = baseline_results.get('metrics', {}).get('average_precision_weak_questions', 0.0)
    baseline_weak_count = baseline_results.get('metrics', {}).get('weak_questions_count', 0)
    print(f"Baseline Weak Questions (P<0.8): {baseline_weak:.4f} ({baseline_weak_count} questions)")
    print()
    
    all_precisions = {
        'Query Expansion': qe_results.get('metrics', {}).get('average_precision_at_k', 0.0),
        'Re-ranking': rr_results.get('metrics', {}).get('average_precision_at_k', 0.0),
        'Hybrid Search': hybrid_results.get('metrics', {}).get('average_precision_at_k', 0.0),
        'Both (QE+RR)': both_results.get('metrics', {}).get('average_precision_at_k', 0.0)
    }
    best_precision = max(all_precisions.values())
    best_method = max(all_precisions, key=all_precisions.get)
    print(f"Best Precision@K: {best_precision:.4f} ({best_method})")
    
    all_weak_precisions = {
        'Query Expansion': qe_results.get('metrics', {}).get('average_precision_weak_questions', 0.0),
        'Re-ranking': rr_results.get('metrics', {}).get('average_precision_weak_questions', 0.0),
        'Hybrid Search': hybrid_results.get('metrics', {}).get('average_precision_weak_questions', 0.0),
        'Both (QE+RR)': both_results.get('metrics', {}).get('average_precision_weak_questions', 0.0)
    }
    best_weak_precision = max(all_weak_precisions.values())
    best_weak_method = max(all_weak_precisions, key=all_weak_precisions.get)
    print(f"Best Weak Questions Precision: {best_weak_precision:.4f} ({best_weak_method})")
    
    best_improvement = max(
        comparisons.get('query_expansion', {}).get('improvements', {}).get('precision_improvement_percent', 0.0),
        comparisons.get('reranking', {}).get('improvements', {}).get('precision_improvement_percent', 0.0),
        comparisons.get('hybrid', {}).get('improvements', {}).get('precision_improvement_percent', 0.0),
        comparisons.get('both', {}).get('improvements', {}).get('precision_improvement_percent', 0.0)
    )
    
    best_weak_improvement = max(
        comparisons.get('query_expansion', {}).get('improvements', {}).get('weak_precision_improvement_percent', 0.0),
        comparisons.get('reranking', {}).get('improvements', {}).get('weak_precision_improvement_percent', 0.0),
        comparisons.get('hybrid', {}).get('improvements', {}).get('weak_precision_improvement_percent', 0.0),
        comparisons.get('both', {}).get('improvements', {}).get('weak_precision_improvement_percent', 0.0)
    )
    
    print(f"Best Overall Precision Improvement: {best_improvement:+.2f}%")
    print(f"Best Weak Questions Precision Improvement: {best_weak_improvement:+.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

