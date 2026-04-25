import torch
import numpy as np
import pandas as pd
import pickle
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

class StudyChatEmbedder:
    def __init__(self, model_name='BAAI/bge-m3', use_fp16=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing StudyChatEmbedder on: {self.device} ---")
        

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=self.device)
        print(f"--- BGE-M3 Transformer loaded ---")

    def get_embeddings_and_similarity(self, submission_text, dialogue_turns):
        """
        Calculates embeddings and the max cosine similarity.
        Returns: sub_vec (1, 1024), diag_vecs (n, 1024), max_sim (float)
        """
        if not dialogue_turns or not submission_text:
            return None, None, 0.0

        # 1. Encode Submission
        sub_res = self.model.encode([submission_text], batch_size=1, max_length=8192)
        sub_vec = np.array(sub_res['dense_vecs'])

        # 2. Encode all dialogue turns
        diag_res = self.model.encode(dialogue_turns, batch_size=len(dialogue_turns), max_length=8192)
        diag_vecs = np.array(diag_res['dense_vecs'])

        # 3. Compute Cosine Similarities
        similarities = cosine_similarity(sub_vec, diag_vecs)
        max_sim = float(np.max(similarities))
        
        return sub_vec, diag_vecs, max_sim

def compute_similarity_features(df_dialogue, df_submission, save_vectors=True):
    embedder = StudyChatEmbedder()
    results = []
    vector_storage = {}
    
    # 1. PRE-ENCODE EVERYTHING IN BULK
    # Get all unique dialogue turns and submissions across the dataset
    all_dialogue_text = [item for sublist in df_dialogue['dialogue_pairs'] for item in sublist]
    all_submission_text = df_submission['submission_content'].tolist()
    
    print(f"--- Encoding {len(all_dialogue_text)} dialogue turns in bulk ---")
    # Batch size 16/32 is safer for 6GB VRAM with BGE-M3
    diag_res = embedder.model.encode(all_dialogue_text, batch_size=32, max_length=512)
    diag_all_vecs = torch.tensor(diag_res['dense_vecs']).to(embedder.device)
    
    print(f"--- Encoding {len(all_submission_text)} submissions in bulk ---")
    sub_res = embedder.model.encode(all_submission_text, batch_size=16, max_length=8192)
    sub_all_vecs = torch.tensor(sub_res['dense_vecs']).to(embedder.device)

    # 2. FAST LOOKUP
    # Create a mapping so we can find the vectors without re-encoding
    diag_ptr = 0
    common_indices = df_dialogue.index.intersection(df_submission.index)

    for idx in common_indices:
        user_id, assignment_id = idx
        
        # Grab pre-computed submission vector
        s_pos = df_submission.index.get_loc(idx)
        sub_v = sub_all_vecs[s_pos].view(1, -1)
        
        # Grab pre-computed dialogue vectors for this specific student/assignment
        num_turns = len(df_dialogue.loc[idx, 'dialogue_pairs'])
        diag_vs = diag_all_vecs[diag_ptr : diag_ptr + num_turns]
        diag_ptr += num_turns

        # 3. GPU SIMILARITY
        similarities = torch.mm(sub_v, diag_vs.t())
        max_sim = float(torch.max(similarities))
        final_vecs = np.array([v["submission_vector"] for v in vector_storage.values()]) if vector_storage else np.array([])

        results.append({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "max_similarity_score": max_sim
        })

        if save_vectors:
            vector_storage[f"{user_id}_{assignment_id}"] = {
                "submission_vector": sub_v.cpu().numpy(),
                "dialogue_vectors": diag_vs.cpu().numpy()
            }


    return pd.DataFrame(results) , final_vecs