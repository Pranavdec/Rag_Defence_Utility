"""
MBA (Membership Inference Attack) Experiments Runner

This script runs membership inference attacks against the RAG system to measure
attack success rates across different defense configurations.

Usage:
    python scripts/run_mba_experiments.py [dataset] --M [num_masks] --gamma [threshold]
    
Examples:
    python scripts/run_mba_experiments.py nq
    python scripts/run_mba_experiments.py pubmedqa --M 15 --gamma 0.6
"""

import argparse
import os
import sys
import json
import yaml
import random
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attacks.mba import MBAFramework
from src.core.retrieval import VectorStore
from src.core.generation import create_generator
from src.defenses.manager import DefenseManager
from src.data_loaders.nq_loader import NQLoader
from src.data_loaders.pubmed_loader import PubMedLoader
from src.data_loaders.trivia_loader import TriviaLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def load_config():
    """Load configuration from YAML file."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def get_loader(name: str):
    """Get the appropriate data loader."""
    loaders = {
        "nq": NQLoader,
        "pubmedqa": PubMedLoader,
        "triviaqa": TriviaLoader,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name]()


class SimpleRAGWrapper:
    """Wrapper to make RAG components compatible with MBA attack interface."""
    
    def __init__(self, vector_store, generator, defense_manager, top_k: int = 5):
        self.vector_store = vector_store
        self.generator = generator
        self.defense_manager = defense_manager
        self.top_k = top_k
    
    def run_single(self, question: str) -> Dict:
        """
        Run RAG pipeline for a single question.
        
        Args:
            question: Query text
            
        Returns:
            Dict with 'answer' key
        """
        # Apply pre-retrieval defenses
        query_text, fetch_k = self.defense_manager.apply_pre_retrieval(question, self.top_k)
        
        # Retrieve documents
        retrieved = self.vector_store.query(query_text, top_k=fetch_k)
        
        # Apply post-retrieval defenses
        retrieved = self.defense_manager.apply_post_retrieval(retrieved, question)
        
        # Extract contexts
        contexts = [r["content"] for r in retrieved]
        
        # Generate answer
        answer = self.generator.generate(question, contexts)
        
        return {'answer': answer}


def chunk_text(text: str, size: int, overlap: int) -> list:
    """Simple text chunking - matches ingestion logic."""
    if len(text) <= size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap if overlap < end else end
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Run MBA membership inference experiments")
    parser.add_argument("dataset", nargs="?", default="nq", 
                        choices=["nq", "pubmedqa", "triviaqa"],
                        help="Dataset to attack (default: nq)")
    parser.add_argument("--M", type=int, 
                        help="Number of masks per document (default: from config)")
    parser.add_argument("--gamma", type=float,
                        help="Accuracy threshold for membership (default: from config)")
    parser.add_argument("--num-members", type=int,
                        help="Number of member documents to test (default: from config)")
    parser.add_argument("--num-non-members", type=int,
                        help="Number of non-member documents to test (default: from config)")
    parser.add_argument("--device", type=str,
                        help="Device for models: cuda, cpu, or auto (default: from config)")
    parser.add_argument("--proxy-model", type=str,
                        help="Proxy model for difficulty scoring (default: from config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Get MBA config with fallback defaults
    mba_config = config.get('attack', {}).get('mba', {})
    
    # Use config values as defaults, allow CLI args to override
    M = args.M if args.M is not None else mba_config.get('M', 10)
    gamma = args.gamma if args.gamma is not None else mba_config.get('gamma', 0.5)
    num_members = args.num_members if args.num_members is not None else mba_config.get('num_members', 50)
    num_non_members = args.num_non_members if args.num_non_members is not None else mba_config.get('num_non_members', 50)
    device = args.device if args.device is not None else mba_config.get('device', 'auto')
    proxy_model = args.proxy_model if args.proxy_model is not None else mba_config.get('proxy_model', 'gpt2-xl')
    enable_spelling = mba_config.get('enable_spelling_correction', False)
    max_document_words = mba_config.get('max_document_words', 500)
    
    print("=" * 70)
    print("MBA MEMBERSHIP INFERENCE ATTACK EXPERIMENTS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Masks per document (M): {M}")
    print(f"Membership threshold (γ): {gamma}")
    print(f"Members to test: {num_members}")
    print(f"Non-members to test: {num_non_members}")
    print(f"Proxy model: {proxy_model}")
    print(f"Spelling correction: {enable_spelling}")
    print(f"Device: {device}")
    print("=" * 70)
    
    ingestion_seed = config["data"]["ingestion_seed"]
    ingestion_size = config["data"]["ingestion_size"]
    
    print(f"\nConfiguration:")
    print(f"  Ingestion seed: {ingestion_seed}")
    print(f"  Ingestion size: {ingestion_size}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    loader = get_loader(args.dataset)
    all_qa_pairs = loader.load_qa_pairs(limit=None)
    print(f"  Total QA pairs available: {len(all_qa_pairs)}")
    
    # Replicate the ingestion sampling to identify member documents
    print(f"\nIdentifying member vs non-member documents...")
    random.seed(ingestion_seed)
    
    if len(all_qa_pairs) > ingestion_size:
        member_qa_pairs = random.sample(all_qa_pairs, ingestion_size)
    else:
        member_qa_pairs = all_qa_pairs
    
    member_ids = {qa.pair_id for qa in member_qa_pairs}
    print(f"  Member QA pairs (in vector store): {len(member_ids)}")
    
    # Get non-member documents
    non_member_qa_pairs = [qa for qa in all_qa_pairs if qa.pair_id not in member_ids]
    print(f"  Non-member QA pairs (not in vector store): {len(non_member_qa_pairs)}")
    
    if len(non_member_qa_pairs) == 0:
        print("\n✗ Error: No non-member documents available!")
        print("  The entire dataset was ingested. Cannot perform membership inference.")
        print("  Suggestion: Reduce ingestion_size in config.yaml or use a larger dataset.")
        return
    
    # Create chunks from member and non-member documents (same chunking as ingestion)
    chunk_size = config["retrieval"]["chunk_size"]
    chunk_overlap = config["retrieval"]["chunk_overlap"]
    
    print(f"\nCreating chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    member_chunks = []
    for qa in member_qa_pairs:
        for passage in qa.gold_passages:
            chunks = chunk_text(passage, chunk_size, chunk_overlap)
            for chunk in chunks:
                member_chunks.append({'text': chunk, 'pair_id': qa.pair_id})
    
    non_member_chunks = []
    for qa in non_member_qa_pairs:
        for passage in qa.gold_passages:
            chunks = chunk_text(passage, chunk_size, chunk_overlap)
            for chunk in chunks:
                non_member_chunks.append({'text': chunk, 'pair_id': qa.pair_id})
    
    print(f"  Member chunks: {len(member_chunks)}")
    print(f"  Non-member chunks: {len(non_member_chunks)}")
    
    # Sample test sets from chunks
    num_members_actual = min(num_members, len(member_chunks))
    num_non_members_actual = min(num_non_members, len(non_member_chunks))
    
    random.seed(42)  # Use different seed for test sampling
    test_member_chunks = random.sample(member_chunks, num_members_actual)
    test_non_member_chunks = random.sample(non_member_chunks, num_non_members_actual)
    
    print(f"\nTest set:")
    print(f"  Member chunks to test: {len(test_member_chunks)}")
    print(f"  Non-member chunks to test: {len(test_non_member_chunks)}")
    
    # Initialize RAG system
    print(f"\nInitializing RAG system...")
    
    # Vector store
    vector_store = VectorStore(
        collection_name=loader.name,
        persist_directory=config["paths"]["chroma_db"],
        embedding_model=config["system"]["embedding_model"]
    )
    
    if not vector_store.is_populated():
        print(f"✗ Error: Vector store for {args.dataset} is not populated!")
        print("  Run 'python scripts/ingest_data.py' first.")
        return
    
    # Defense manager
    defense_config = config.get("defenses", [])
    defense_manager = DefenseManager(defense_config)
    
    # Generator
    generator = create_generator(config, defense_manager=defense_manager)
    
    # Wrap RAG components
    top_k = config["retrieval"]["top_k"]
    rag_system = SimpleRAGWrapper(vector_store, generator, defense_manager, top_k)
    
    print(f"  Vector store: {loader.name}")
    print(f"  Top-k retrieval: {top_k}")
    print(f"  Active defenses: {[d['name'] for d in defense_config if d.get('enabled', False)]}")
    
    # Initialize MBA Framework
    print(f"\nInitializing MBA Framework...")
    mba = MBAFramework(
        M=M, 
        gamma=gamma, 
        device=device,
        proxy_model=proxy_model,
        enable_spelling=enable_spelling,
        max_document_words=max_document_words
    )
    
    # Run attacks
    print(f"\n" + "=" * 70)
    print("RUNNING ATTACKS")
    print("=" * 70)
    
    results = {
        'config': {
            'dataset': args.dataset,
            'M': M,
            'gamma': gamma,
            'proxy_model': proxy_model,
            'enable_spelling': enable_spelling,
            'device': device,
            'ingestion_seed': ingestion_seed,
            'ingestion_size': ingestion_size,
            'num_test_members': len(test_member_chunks),
            'num_test_non_members': len(test_non_member_chunks),
            'top_k': top_k,
            'defenses': defense_config,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'timestamp': datetime.now().isoformat()
        },
        'member_results': [],
        'non_member_results': []
    }
    
    # Attack member chunks
    print(f"\nAttacking MEMBER chunks...")
    for idx, chunk_data in enumerate(tqdm(test_member_chunks, desc="Members")):
        target_chunk = chunk_data['text']
        
        try:
            attack_result = mba.attack(target_chunk, rag_system)
            attack_result['pair_id'] = chunk_data['pair_id']
            attack_result['is_actual_member'] = True
            attack_result['chunk_length'] = len(target_chunk)
            results['member_results'].append(attack_result)
        except Exception as e:
            logger.error(f"Attack failed on member chunk {idx}: {e}")
            results['member_results'].append({
                'pair_id': chunk_data['pair_id'],
                'is_actual_member': True,
                'error': str(e)
            })
    
    # Attack non-member chunks
    print(f"\nAttacking NON-MEMBER chunks...")
    for idx, chunk_data in enumerate(tqdm(test_non_member_chunks, desc="Non-members")):
        target_chunk = chunk_data['text']
        
        try:
            attack_result = mba.attack(target_chunk, rag_system)
            attack_result['pair_id'] = chunk_data['pair_id']
            attack_result['is_actual_member'] = False
            attack_result['chunk_length'] = len(target_chunk)
            results['non_member_results'].append(attack_result)
        except Exception as e:
            logger.error(f"Attack failed on non-member chunk {idx}: {e}")
            results['non_member_results'].append({
                'pair_id': chunk_data['pair_id'],
                'is_actual_member': False,
                'error': str(e)
            })
    
    # Calculate metrics
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Member metrics
    member_predictions = [r.get('is_member', False) for r in results['member_results'] if 'error' not in r]
    member_accuracies = [r.get('accuracy', 0) for r in results['member_results'] if 'error' not in r]
    
    # Non-member metrics
    non_member_predictions = [r.get('is_member', False) for r in results['non_member_results'] if 'error' not in r]
    non_member_accuracies = [r.get('accuracy', 0) for r in results['non_member_results'] if 'error' not in r]
    
    # Calculate attack metrics
    true_positives = sum(member_predictions)  # Correctly identified members
    false_negatives = len(member_predictions) - true_positives
    false_positives = sum(non_member_predictions)  # Incorrectly identified non-members as members
    true_negatives = len(non_member_predictions) - false_positives
    
    total_correct = true_positives + true_negatives
    total_predictions = len(member_predictions) + len(non_member_predictions)
    
    attack_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    tpr = true_positives / len(member_predictions) if len(member_predictions) > 0 else 0  # True Positive Rate
    fpr = false_positives / len(non_member_predictions) if len(non_member_predictions) > 0 else 0  # False Positive Rate
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    print(f"\nAttack Success Metrics:")
    print(f"  Attack Accuracy: {attack_accuracy:.3f} ({total_correct}/{total_predictions})")
    print(f"  True Positive Rate (Recall): {tpr:.3f}")
    print(f"  False Positive Rate: {fpr:.3f}")
    print(f"  Precision: {precision:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {true_positives:4d} (member correctly identified)")
    print(f"  False Negatives: {false_negatives:4d} (member missed)")
    print(f"  False Positives: {false_positives:4d} (non-member incorrectly identified)")
    print(f"  True Negatives:  {true_negatives:4d} (non-member correctly identified)")
    
    print(f"\nPrediction Accuracy (for filled masks):")
    print(f"  Members avg: {sum(member_accuracies)/len(member_accuracies):.3f}" if member_accuracies else "  Members avg: N/A")
    print(f"  Non-members avg: {sum(non_member_accuracies)/len(non_member_accuracies):.3f}" if non_member_accuracies else "  Non-members avg: N/A")
    
    # Add summary to results
    results['summary'] = {
        'attack_accuracy': float(attack_accuracy),
        'true_positive_rate': float(tpr),
        'false_positive_rate': float(fpr),
        'precision': float(precision),
        'true_positives': int(true_positives),
        'false_negatives': int(false_negatives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'member_avg_accuracy': float(sum(member_accuracies)/len(member_accuracies)) if member_accuracies else 0,
        'non_member_avg_accuracy': float(sum(non_member_accuracies)/len(non_member_accuracies)) if non_member_accuracies else 0
    }
    
    # Save results
    os.makedirs("data/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/mba_{args.dataset}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
