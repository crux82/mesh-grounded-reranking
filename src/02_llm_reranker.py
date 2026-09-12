import os
import json
import re
import argparse
from tqdm import tqdm

# Import vLLM
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Import openai_harmony
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding, Conversation,
    Message, Role, SystemContent, DeveloperContent, ReasoningEffort
)

# --- Config ---
MODEL_NAME = "openai/gpt-oss-20b"
ALPHA_WEIGHT = 0.3
DEF_NUMBER = 5
REASONING = 'Medium'

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def format_mesh_with_definitions(mesh_list, definitions_dict, def_number=5):
    if not mesh_list or def_number == 0: 
        return "None available."
    
    context_parts = []
    top_mesh = mesh_list[:def_number]
    
    for item in top_mesh:
        raw_term = item.get('term', '')
        clean_term = raw_term.replace("_", " ")
        definition = definitions_dict.get(clean_term, "Definition not available.")
        
        raw_children = item.get('narrower', [])
        child_details = []
        for c in raw_children:
            def_c = definitions_dict.get(c, "Definition not available.")
            child_details.append(f"{c}: {def_c}")
            
        entry = f"- **{clean_term}**: {definition}"
        if child_details:
            entry += f"\n  -> Specific Subtypes (Definitions): {'; '.join(child_details)}"
        
        context_parts.append(entry)
            
    return "\n".join(context_parts)

def create_rerank_conversation(prompt_template, query_text, mesh_context_str, doc_data, reasoning_effort) -> Conversation:
    
    abstract_text = doc_data.get("abstract_text") or doc_data.get("body") or doc_data.get("text", "")
    title = doc_data.get("article_title", "")
    
    # Concat title and abstract
    full_abstract_content = f"{title} {abstract_text}".strip()
    
    formatted_user_content = prompt_template.format(
        question=query_text,
        mesh_context=mesh_context_str,
        abstract=full_abstract_content
    )
    
    if reasoning_effort == 'Medium':
        r_effort = ReasoningEffort.MEDIUM
    elif reasoning_effort == 'High':
        r_effort = ReasoningEffort.HIGH
    else:
        r_effort = ReasoningEffort.LOW
        
    sys_msg = Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new().with_reasoning_effort(r_effort)
    )
    
    dev_msg = Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions("You are a helpful biomedical QA assistant.")
    )

    user_msg = Message.from_role_and_content(Role.USER, formatted_user_content)

    return Conversation.from_messages([sys_msg, dev_msg, user_msg])

def llm_rerank_batch(llm_engine: LLM, encoding, prompt_template, query_text, mesh_context_str, documents, reasoning_effort, max_tokens) -> list[float]:
    prompts_tokens = []
    
    for doc in documents:
        convo = create_rerank_conversation(prompt_template, query_text, mesh_context_str, doc, reasoning_effort)
        token_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompts_tokens.append(TokensPrompt(prompt_token_ids=token_ids))

    if not prompts_tokens:
        return []

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm_engine.generate(prompts_tokens, sampling_params, use_tqdm=False)
    
    scores = []
    for output in outputs:
        generated_text = output.outputs[0].text
        match = re.search(r"assistantfinal\s*([0-9\.]+)", generated_text)
        if match:
            try:
                val = float(match.group(1))
                scores.append(max(0.0, min(1.0, val)))
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate Prompts and Rerank with vLLM.")
    parser.add_argument("--batch", type=str, required=True, help="Batch number to process")
    args = parser.parse_args()
    
    batch_num = args.batch
    
    # 1. Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "prompts", "system_prompt.txt")
    scope_notes_path = os.path.join(base_dir, "data", "mesh", "mesh_scope_notes.json")
    
    input_dir = os.path.join(base_dir, "data", "02_mesh_grounded_batches")
    input_file = os.path.join(input_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_mesh_grounded.json")
    
    output_dir = os.path.join(base_dir, "data", "03_scored_batches")
    output_file = os.path.join(output_dir, f"BioASQ-task13bPhaseB-testset{batch_num}_scored.json")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Caricamento Risorse Statiche
    print("Loading prompt template and MeSH scope notes...")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    scope_notes = load_json(scope_notes_path)
    
    print(f"Loading mesh-grounded batch {batch_num}...")
    queries = load_json(input_file)
    
    # 3. Init vLLM and Encoding 
    print(f"Loading vLLM Model: {MODEL_NAME}...")
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, dtype="auto", trust_remote_code=True, gpu_memory_utilization=0.75)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    max_tokens = 1536 + 256 if REASONING == 'Medium' else 512
    
    # 4. Reranking Loop
    print(f"Starting vLLM Reranking (alpha = {ALPHA_WEIGHT})...")
    for q in tqdm(queries, desc=f"Reranking Batch {batch_num}"):
        query_text = q.get("body", "")
        mesh_terms = q.get("retrieved_mesh_terms", [])
        
        # Formatting Mesh context
        mesh_context_str = format_mesh_with_definitions(mesh_terms, scope_notes, def_number=DEF_NUMBER)
        
        candidates = q.get("candidates", [])
        if not candidates:
            continue
            
        # Batch inference
        llm_scores = llm_rerank_batch(
            llm_engine=llm, 
            encoding=encoding, 
            prompt_template=prompt_template, 
            query_text=query_text, 
            mesh_context_str=mesh_context_str, 
            documents=candidates, 
            reasoning_effort=REASONING, 
            max_tokens=max_tokens
        )
        
        # scores update and final interpolation
        for idx, doc in enumerate(candidates):
            base_score = doc.get("base_score", 0.0)
            s_llm = llm_scores[idx] if idx < len(llm_scores) else 0.0
            
            final_score = (ALPHA_WEIGHT * s_llm) + ((1 - ALPHA_WEIGHT) * base_score)
            
            doc["llm_score"] = s_llm
            doc["final_score"] = final_score
            
        # final sorting
        q["candidates"] = sorted(candidates, key=lambda x: x.get("final_score", 0.0), reverse=True)

    print(f"Saving final scored results to {output_file}...")
    save_json(queries, output_file)

if __name__ == "__main__":
    main()