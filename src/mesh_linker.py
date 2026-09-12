import json
import numpy as np

class MeSHLinker:
    def __init__(self, mesh_emb_path, hierarchy_path, threshold=0.45, top_k=5):
        self.threshold = threshold
        self.top_k = top_k
        
        # 1. Load Embedding
        with open(mesh_emb_path, 'r', encoding='utf-8') as f:
            self.mesh_embeddings = json.load(f)
        self.mesh_terms = list(self.mesh_embeddings.keys())
        
        # 2. Load Hyerarchy (Narrower terms)
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            self.narrower_dict = json.load(f).get("broader_as_key", {})
            
        # 3. Init Matrix
        mesh_matrix = np.array([self.mesh_embeddings[t] for t in self.mesh_terms], dtype=np.float32)
        self.mesh_matrix_norm = self._normalize_matrix(mesh_matrix)

    def _normalize_matrix(self, matrix):
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1 
        return matrix / norm

    def _normalize_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_top_concepts(self, query_sbert_vector):
        """
        Compute cosine similarity and return a list of dictionaries with top-K concepts and narrower terms.
    
        """
        q_vec_norm = self._normalize_vector(query_sbert_vector)
        scores = np.dot(self.mesh_matrix_norm, q_vec_norm)
        
        valid_indices = np.where(scores >= self.threshold)[0]
        if len(valid_indices) == 0:
            return [] # No concept above thresh
            
        valid_scores = scores[valid_indices]
        sorted_valid_positions = np.argsort(valid_scores)[::-1][:self.top_k]
        top_k_indices = valid_indices[sorted_valid_positions]
        
        retrieved_concepts = []
        for idx in top_k_indices:
            term = self.mesh_terms[idx]
            term_for_search = term.replace("_", " ")
            narrower_terms = self.narrower_dict.get(term_for_search, [])
            
            retrieved_concepts.append({
                "term": term,
                "narrower": narrower_terms
            })
            
        return retrieved_concepts