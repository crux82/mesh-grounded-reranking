import os
import json
import argparse
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
K_VALUES = [3, 5, 10]

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_difficulty_class(recall_score):
    if recall_score == 0.0:
        return "Not Threaded (0%)"
    elif 0.0 < recall_score <= 0.20:
        return "Hard (0-20%)"
    elif 0.20 < recall_score <= 0.70:
        return "Medium (20-70%)"
    elif 0.70 < recall_score <= 1.00:
        return "Simple (70-100%)"
    return "Unknown"

def calculate_ap(retrieved_ids, gt_docs, k):
    """Compute l'Average Precision a K"""
    if not gt_docs: return 0.0
    retrieved_k = retrieved_ids[:k]
    ap = 0.0
    rel_cnt = 0
    for i, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in gt_docs:
            rel_cnt += 1
            ap += rel_cnt / i
    divisor = min(k, len(gt_docs)) if min(k, len(gt_docs)) > 0 else 1
    return ap / divisor

def main():
    parser = argparse.ArgumentParser(description="Step 3: Evaluate MAP for multiple K and query difficulties.")
    parser.add_argument("--batch", type=str, required=True, help="Batch number to process")
    args = parser.parse_args()
    
    batch_num = args.batch
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "data", "03_scored_batches")
    input_file = os.path.join(input_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_scored.json")
    
    output_dir = os.path.join(base_dir, "data", "04_evaluation")
    output_file = os.path.join(output_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_metrics.json")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading scored batch {batch_num} from {input_file}...")
    queries = load_json(input_file)
    
   
    # metrics_data[k][class] = { 'ap_hybrid': [], 'ap_final': [] }
    metrics_data = {k: defaultdict(lambda: {'ap_hybrid': [], 'ap_final': []}) for k in K_VALUES}
    
    global_counts = defaultdict(int)
    
    for q in queries:
        gt_docs = set([str(doc_id) for doc_id in q.get("documents", [])])
        if not gt_docs:
            continue
            
        candidates = q.get("candidates", [])
        
        # 1. Sort per BASE SCORE (Hybrid Retrieval)
        sorted_hybrid = sorted(candidates, key=lambda x: x.get("base_score", 0.0), reverse=True)
        ids_hybrid = [str(c.get("doc_id", "")) for c in sorted_hybrid]
        
        # 2. Compute Recall@50 to establish query complexity
        ids_top50_hybrid = ids_hybrid[:50]
        relevant_retrieved = len(set(ids_top50_hybrid) & gt_docs)
        recall_50 = relevant_retrieved / len(gt_docs) if len(gt_docs) > 0 else 0.0
        q_class = get_difficulty_class(recall_50)
        global_counts[q_class] += 1
        
        # 3. Sort per FINAL SCORE (Interpolated LLM)
        sorted_final = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)
        ids_final = [str(c.get("doc_id", "")) for c in sorted_final]
        
        
        for k in K_VALUES:
            ap_hybrid = calculate_ap(ids_hybrid, gt_docs, k)
            ap_final = calculate_ap(ids_final, gt_docs, k)
            
            # Salviamo sia per la singola classe che per il totale Globale
            metrics_data[k][q_class]['ap_hybrid'].append(ap_hybrid)
            metrics_data[k][q_class]['ap_final'].append(ap_final)
            
            metrics_data[k]['GLOBAL']['ap_hybrid'].append(ap_hybrid)
            metrics_data[k]['GLOBAL']['ap_final'].append(ap_final)

    class_order = ["Not Threaded (0%)", "Hard (0-20%)", "Medium (20-70%)", "Simple (70-100%)", "GLOBAL"]
    
    json_results = {"batch": batch_num, "results": {}}

    print("\n" + "="*85)
    print(f" EVALUATION RESULTS - BATCH {batch_num}")
    print("="*85)

    for k in K_VALUES:
        print(f"\n>>> MAP@{k} ANALYSIS")
        print(f"{'CLASS':<20} | {'COUNT':<5} | {'HYBRID':<7} | {'LLM FINAL':<9} | {'DELTA':<7}")
        print("-" * 60)
        
        json_results["results"][f"MAP@{k}"] = {}
        
        for cls in class_order:
            vals = metrics_data[k].get(cls)
            if vals and len(vals['ap_hybrid']) > 0:
                count = len(vals['ap_hybrid'])
                map_hybrid = np.mean(vals['ap_hybrid']) * 100
                map_final = np.mean(vals['ap_final']) * 100
                delta = map_final - map_hybrid
                
                if cls == "GLOBAL":
                    print("-" * 60)
                
                print(f"{cls:<20} | {count:<5} | {map_hybrid:.2f}   | {map_final:.2f}     | {delta:+.2f}")
                
                json_results["results"][f"MAP@{k}"][cls] = {
                    "count": count,
                    "map_hybrid": float(map_hybrid),
                    "map_final": float(map_final),
                    "delta": float(delta)
                }
            else:
                print(f"{cls:<20} | 0     | N/A     | N/A       | N/A")

    print("\n" + "="*85)
    print(f"Saving detailed metrics to {output_file}...")
    save_json(json_results, output_file)

if __name__ == "__main__":
    main()