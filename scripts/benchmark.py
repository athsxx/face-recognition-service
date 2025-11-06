"""Benchmarking and evaluation script for FRS."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from frs.core.alignment import FaceAligner
from frs.core.detector import FaceDetector
from frs.core.embedding import FaceEmbedding
from frs.core.matcher import FaceMatcher
from frs.database.models import Database


class Benchmark:
    """Benchmark FRS components."""

    def __init__(self):
        """Initialize benchmark."""
        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedding()
        
        # Initialize database and matcher
        self.db = Database("sqlite:///data/benchmark.db")
        self.db.create_tables()
        self.matcher = FaceMatcher(self.db)

    def benchmark_detection(self, image_dir: Path, num_images: int = 100) -> Dict:
        """Benchmark face detection.

        Args:
            image_dir: Directory containing test images
            num_images: Number of images to test

        Returns:
            Benchmark results
        """
        logger.info("Benchmarking face detection...")
        
        images = list(image_dir.glob("*.jpg"))[:num_images]
        
        latencies = []
        num_faces_detected = []
        
        for img_path in tqdm(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            start = time.time()
            faces = self.detector.detect(img)
            latency = (time.time() - start) * 1000
            
            latencies.append(latency)
            num_faces_detected.append(len(faces))
        
        results = {
            'component': 'detection',
            'num_images': len(latencies),
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'throughput_fps': 1000 / np.mean(latencies),
            'avg_faces_per_image': np.mean(num_faces_detected)
        }
        
        return results

    def benchmark_embedding(self, num_iterations: int = 100, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict:
        """Benchmark embedding extraction.

        Args:
            num_iterations: Number of iterations per batch size
            batch_sizes: List of batch sizes to test

        Returns:
            Benchmark results
        """
        logger.info("Benchmarking embedding extraction...")
        
        results = {'component': 'embedding', 'batch_results': []}
        
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = [np.random.randn(112, 112, 3).astype(np.float32) 
                          for _ in range(batch_size)]
            
            # Warmup
            for _ in range(10):
                _ = self.embedder.extract(dummy_input)
            
            # Benchmark
            latencies = []
            for _ in range(num_iterations):
                start = time.time()
                _ = self.embedder.extract(dummy_input)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            throughput = (batch_size * 1000) / avg_latency
            
            results['batch_results'].append({
                'batch_size': batch_size,
                'avg_latency_ms': avg_latency,
                'std_latency_ms': np.std(latencies),
                'throughput_fps': throughput
            })
        
        return results

    def benchmark_matching(self, gallery_size: int = 1000, num_queries: int = 100) -> Dict:
        """Benchmark face matching.

        Args:
            gallery_size: Number of identities in gallery
            num_queries: Number of query embeddings

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking matching with gallery size={gallery_size}...")
        
        # Create dummy gallery
        embedding_dim = 512
        for i in range(gallery_size):
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            
            self.matcher.add_identity(
                identity_id=f"id_{i}",
                name=f"Person_{i}",
                embedding=embedding
            )
        
        # Create query embeddings
        queries = []
        for _ in range(num_queries):
            emb = np.random.randn(embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            queries.append(emb)
        
        # Benchmark
        latencies = []
        for query in tqdm(queries):
            start = time.time()
            _ = self.matcher.match(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        results = {
            'component': 'matching',
            'gallery_size': gallery_size,
            'num_queries': num_queries,
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'throughput_qps': 1000 / np.mean(latencies)
        }
        
        return results

    def evaluate_recognition(self, val_data_path: Path) -> Dict:
        """Evaluate recognition accuracy on validation set.

        Args:
            val_data_path: Path to validation JSON file

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating recognition accuracy...")
        
        # Load validation data
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        
        # Extract embeddings for gallery (use subset for efficiency)
        gallery_identities = {}
        for item in val_data[:100]:  # Limit gallery size
            identity = item['identity']
            if identity not in gallery_identities:
                img_path = item['image_path']
                img = cv2.imread(img_path)
                if img is not None:
                    faces = self.detector.detect(img)
                    if len(faces) > 0:
                        aligned = self.aligner.align(img, faces[0]['landmarks'])
                        embedding = self.embedder.extract(aligned)
                        gallery_identities[identity] = embedding
                        
                        self.matcher.add_identity(
                            identity_id=identity,
                            name=identity,
                            embedding=embedding
                        )
        
        # Test on remaining data
        y_true = []
        y_pred = []
        
        for item in tqdm(val_data[100:200]):  # Test subset
            identity = item['identity']
            img_path = item['image_path']
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            faces = self.detector.detect(img)
            if len(faces) == 0:
                continue
            
            aligned = self.aligner.align(img, faces[0]['landmarks'])
            embedding = self.embedder.extract(aligned)
            
            # Match
            matches = self.matcher.match(embedding)
            
            y_true.append(identity)
            if matches:
                y_pred.append(matches[0]['identity_id'])
            else:
                y_pred.append('unknown')
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Top-1 and Top-5 accuracy
        top1_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        top1_acc = top1_correct / len(y_true)
        
        results = {
            'component': 'recognition',
            'num_gallery': len(gallery_identities),
            'num_test': len(y_true),
            'top1_accuracy': top1_acc,
            'accuracy': accuracy
        }
        
        return results

    def benchmark_end_to_end(self, image_path: Path, num_iterations: int = 50) -> Dict:
        """Benchmark end-to-end pipeline.

        Args:
            image_path: Test image path
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        logger.info("Benchmarking end-to-end pipeline...")
        
        img = cv2.imread(str(image_path))
        
        latencies = {
            'detection': [],
            'alignment': [],
            'embedding': [],
            'matching': [],
            'total': []
        }
        
        for _ in tqdm(range(num_iterations)):
            start_total = time.time()
            
            # Detection
            start = time.time()
            faces = self.detector.detect(img)
            latencies['detection'].append((time.time() - start) * 1000)
            
            if len(faces) == 0:
                continue
            
            # Alignment
            start = time.time()
            aligned = self.aligner.align(img, faces[0]['landmarks'])
            latencies['alignment'].append((time.time() - start) * 1000)
            
            # Embedding
            start = time.time()
            embedding = self.embedder.extract(aligned)
            latencies['embedding'].append((time.time() - start) * 1000)
            
            # Matching
            start = time.time()
            _ = self.matcher.match(embedding)
            latencies['matching'].append((time.time() - start) * 1000)
            
            latencies['total'].append((time.time() - start_total) * 1000)
        
        results = {
            'component': 'end_to_end',
            'num_iterations': num_iterations,
            'latencies': {
                key: {
                    'avg_ms': np.mean(vals),
                    'std_ms': np.std(vals),
                    'min_ms': np.min(vals),
                    'max_ms': np.max(vals)
                }
                for key, vals in latencies.items() if vals
            },
            'total_throughput_fps': 1000 / np.mean(latencies['total'])
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Face Recognition Service")
    parser.add_argument('--component', type=str, default='all',
                       choices=['all', 'detection', 'embedding', 'matching', 'recognition', 'e2e'],
                       help='Component to benchmark')
    parser.add_argument('--image_dir', type=str, default='data/processed',
                       help='Directory with test images')
    parser.add_argument('--val_data', type=str, default='data/val.json',
                       help='Validation data JSON file')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    benchmark = Benchmark()
    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'results': []}
    
    if args.component in ['all', 'detection']:
        res = benchmark.benchmark_detection(Path(args.image_dir))
        results['results'].append(res)
        logger.info(f"Detection: {res['avg_latency_ms']:.2f}ms, {res['throughput_fps']:.2f} FPS")
    
    if args.component in ['all', 'embedding']:
        res = benchmark.benchmark_embedding()
        results['results'].append(res)
        logger.info("Embedding benchmark complete")
    
    if args.component in ['all', 'matching']:
        res = benchmark.benchmark_matching()
        results['results'].append(res)
        logger.info(f"Matching: {res['avg_latency_ms']:.2f}ms")
    
    if args.component in ['all', 'e2e']:
        test_img = list(Path(args.image_dir).glob("*/*.jpg"))[0]
        res = benchmark.benchmark_end_to_end(test_img)
        results['results'].append(res)
        logger.info(f"End-to-end: {res['latencies']['total']['avg_ms']:.2f}ms")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {args.output}")


if __name__ == '__main__':
    main()
