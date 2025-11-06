#!/usr/bin/env python3
"""
Simple benchmark script for the Face Recognition Service.
"""

import time
import requests
import json
import statistics
from pathlib import Path

def benchmark_api_endpoint(endpoint_url, files=None, data=None, iterations=10):
    """Benchmark an API endpoint."""
    latencies = []
    
    print(f"🔄 Benchmarking {endpoint_url} ({iterations} iterations)...")
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            if files:
                response = requests.post(endpoint_url, files=files, data=data)
            else:
                response = requests.get(endpoint_url)
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                latencies.append(latency)
                print(f"  ✅ Iteration {i+1}: {latency:.1f}ms")
            else:
                print(f"  ❌ Iteration {i+1}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Iteration {i+1}: {str(e)}")
    
    if latencies:
        return {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'throughput_rps': 1000 / statistics.mean(latencies),
            'success_rate': len(latencies) / iterations * 100
        }
    else:
        return None

def main():
    """Run comprehensive benchmark."""
    
    base_url = "http://localhost:8000"
    image_path = "/Users/a91788/Downloads/IMG_1869.jpg"
    
    print("🚀 FACE RECOGNITION SERVICE BENCHMARK")
    print("=" * 50)
    
    # Check if service is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Service not running. Start with: uvicorn frs.api.main:app --host 0.0.0.0 --port 8000")
            return
        print("✅ Service is running")
    except:
        print("❌ Cannot connect to service")
        return
    
    results = {}
    
    # 1. Health Check Benchmark
    print("\n1️⃣ HEALTH CHECK")
    results['health'] = benchmark_api_endpoint(f"{base_url}/health", iterations=20)
    
    # 2. Face Detection Benchmark
    print("\n2️⃣ FACE DETECTION")
    if Path(image_path).exists():
        with open(image_path, 'rb') as f:
            files = {'file': f}
            results['detection'] = benchmark_api_endpoint(
                f"{base_url}/detect", 
                files={'file': open(image_path, 'rb')}, 
                iterations=10
            )
    else:
        print(f"❌ Image not found: {image_path}")
        results['detection'] = None
    
    # 3. Add Identity Benchmark
    print("\n3️⃣ ADD IDENTITY")
    if Path(image_path).exists():
        results['add_identity'] = benchmark_api_endpoint(
            f"{base_url}/add_identity",
            files={'file': open(image_path, 'rb')},
            data={'name': 'Benchmark User', 'identity_id': f'bench_{int(time.time())}'},
            iterations=5
        )
    else:
        results['add_identity'] = None
    
    # 4. Recognition Benchmark
    print("\n4️⃣ FACE RECOGNITION")
    if Path(image_path).exists():
        results['recognition'] = benchmark_api_endpoint(
            f"{base_url}/recognize",
            files={'file': open(image_path, 'rb')},
            data={'return_top_k': '5', 'min_confidence': '0.6'},
            iterations=10
        )
    else:
        results['recognition'] = None
    
    # 5. List Identities Benchmark
    print("\n5️⃣ LIST IDENTITIES")
    results['list_identities'] = benchmark_api_endpoint(f"{base_url}/list_identities", iterations=20)
    
    # Print Summary
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    for endpoint, metrics in results.items():
        if metrics:
            print(f"\n🔹 {endpoint.upper().replace('_', ' ')}")
            print(f"   Mean Latency: {metrics['mean_ms']:.1f}ms")
            print(f"   Median Latency: {metrics['median_ms']:.1f}ms")
            print(f"   Min/Max: {metrics['min_ms']:.1f}ms / {metrics['max_ms']:.1f}ms")
            print(f"   Std Dev: {metrics['std_ms']:.1f}ms")
            print(f"   Throughput: {metrics['throughput_rps']:.1f} RPS")
            print(f"   Success Rate: {metrics['success_rate']:.1f}%")
        else:
            print(f"\n🔹 {endpoint.upper().replace('_', ' ')}: ❌ FAILED")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: benchmark_results.json")
    
    # Performance Analysis
    print("\n🎯 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    if results['detection']:
        det_latency = results['detection']['mean_ms']
        det_fps = 1000 / det_latency
        print(f"🔍 Detection Performance: {det_latency:.1f}ms ({det_fps:.1f} FPS)")
        
        if det_latency < 100:
            print("   ✅ Excellent - Real-time capable")
        elif det_latency < 200:
            print("   ✅ Good - Near real-time")
        elif det_latency < 500:
            print("   ⚠️  Acceptable - Batch processing")
        else:
            print("   ❌ Slow - Needs optimization")
    
    if results['recognition']:
        rec_latency = results['recognition']['mean_ms']
        rec_fps = 1000 / rec_latency
        print(f"👤 Recognition Performance: {rec_latency:.1f}ms ({rec_fps:.1f} FPS)")
        
        if rec_latency < 200:
            print("   ✅ Excellent - Real-time recognition")
        elif rec_latency < 500:
            print("   ✅ Good - Interactive use")
        elif rec_latency < 1000:
            print("   ⚠️  Acceptable - Batch processing")
        else:
            print("   ❌ Slow - Needs optimization")
    
    print("\n🏁 Benchmark Complete!")

if __name__ == "__main__":
    main()