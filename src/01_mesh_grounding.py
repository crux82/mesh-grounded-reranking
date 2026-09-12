import os
import json
import argparse
from tqdm import tqdm
from mesh_linker import MeSHLinker

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Step 1: Ground BioASQ queries with MeSH concepts.")
    parser.add_argument("--batch", type=str, required=True, help="Batch number to process (e.g., 1, 2, 3, 4)")
    args = parser.parse_args()
    
    batch_num = args.batch
    
    # 1. Configurazione dei percorsi aggiornata
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesh_emb_path = os.path.join(base_dir, "data", "mesh", "mesh_embedded_terms.json")
    hierarchy_path = os.path.join(base_dir, "data", "mesh", "mesh_hyerarchy.json")
    
    # Input: 01_vectored_batches
    input_dir = os.path.join(base_dir, "data", "01_vectored_batches")
    input_file = os.path.join(input_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_vectored.json")
    
    # Output: 02_mesh_grounded_batches
    output_dir = os.path.join(base_dir, "data", "02_mesh_grounded_batches")
    output_file = os.path.join(output_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_mesh_grounded.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Inizializzazione MeSH Linker
    print(f"Loading MeSH Linker and vector space...")
    linker = MeSHLinker(
        mesh_emb_path=mesh_emb_path, 
        hierarchy_path=hierarchy_path, 
        threshold=0.45, 
        top_k=5
    )
    
    # 3. Processamento
    print(f"Loading vectored batch {batch_num} from {input_file}...")
    queries = load_json(input_file)
    
    print(f"Grounding {len(queries)} queries with MeSH concepts...")
    for q in tqdm(queries, desc=f"Processing Batch {batch_num}"):
        
        if "dense_vector_SBERT" not in q:
            q["retrieved_mesh_terms"] = []
            continue
            
        q_vec = q["dense_vector_SBERT"]
        retrieved_concepts = linker.get_top_concepts(q_vec)
        q["retrieved_mesh_terms"] = retrieved_concepts

    print(f"Saving mesh-grounded data to {output_file}...")
    save_json(queries, output_file)
    print("Done!")

if __name__ == "__main__":
    main()