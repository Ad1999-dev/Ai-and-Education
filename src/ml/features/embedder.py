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
        
        # use_fp16=True is critical for speed on GPU
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
    vector_storage = {} # Dict to hold raw embeddings

    common_indices = df_dialogue.index.intersection(df_submission.index)

    for idx in common_indices:
        user_id, assignment_id = idx
        
        # Handle index data retrieval
        d_data = df_dialogue.loc[idx]
        dialogue_list = d_data['dialogue_pairs'] if isinstance(d_data, pd.Series) else d_data.iloc[0]['dialogue_pairs']
        
        s_data = df_submission.loc[idx]
        submission_text = s_data['submission_content'] if isinstance(s_data, pd.Series) else s_data.iloc[0]['submission_content']

        print(f"Embedding: {assignment_id} | Student: {user_id[:8]}...")

        # Get the raw vectors and the score
        sub_v, diag_vs, max_sim = embedder.get_embeddings_and_similarity(submission_text, dialogue_list)
        
        # Store metadata and score
        results.append({
            "user_id": user_id,
            "assignment_id": assignment_id,
            "max_similarity_score": max_sim
        })

        # Store the actual 1024-d embeddings
        if save_vectors and sub_v is not None:
            vector_storage[f"{user_id}_{assignment_id}"] = {
                "submission_vector": sub_v,
                "dialogue_vectors": diag_vs
            }

    # Save raw vectors to a pickle file for later regression/analysis
    if save_vectors:
        with open("raw_embeddings.pkl", "wb") as f:
            pickle.dump(vector_storage, f)
        print(f"--- Raw embeddings saved to raw_embeddings.pkl ---")

    return pd.DataFrame(results)