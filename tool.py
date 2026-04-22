from itertools import chain
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from scipy.stats import entropy
from scipy.stats import skew

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd


def DicToJson(dic: dict, path: str):
    """
    Save dict to json file, attempting to convert numeric strings to numbers.
    """
    # Create a copy or transform the data to avoid string-numbers
    cleaned_dic = {k: try_cast(v) for k, v in dic.items()}
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dic, f, ensure_ascii=False, indent=4)

def try_cast(value):
    """Helper to convert string numbers to int or float."""
    if not isinstance(value, str):
        return value
    try:
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_numbers(d):
    for k, v in d.items():
        if isinstance(v, str):
            # Try to convert to int
            try:
                d[k] = int(v)
            except ValueError:
                # Try to convert to float
                try:
                    d[k] = float(v)
                except ValueError:
                    pass # Keep as string if it's not a number
    return d

def JsonToDic(path: str) -> dict:
    """
    Load a json file into a dict
    
    Args :
        path : str
    
    Returns :
        dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=parse_numbers)
    
def get_parasite_word(co_occurrence_df:pd.DataFrame,
                    percentile_threshold:int=95)->tuple[list[str], pd.Series]:
    """Identify 'parasitic' words based on entropy of their co-occurrence distributions.
    co_occurrence_df must have a zero diagonal (no self-cooccurrence).
    Args:
        co_occurrence_df: DataFrame where rows and columns are words, and values are co-occurrence counts.
        percentile_threshold: Percentile threshold to classify words as parasitic.
    Returns:
        bad_words: List of words classified as parasitic.
        scores_series: Series of entropy scores for all words.
    """
    parasitic_scores = {}

    for word, row in co_occurrence_df.iterrows():
        row_vec = row.values
        row_sum = row_vec.sum()
        if row_sum == 0:
            parasitic_scores[word] = 0
            continue
        probs = row_vec / row_sum
        score = entropy(probs, base=len(probs))
        parasitic_scores[word] = score
    scores_series = pd.Series(parasitic_scores).sort_values(ascending=False)
    cutoff_value = np.percentile(scores_series, percentile_threshold)
    bad_words = scores_series[scores_series >= cutoff_value].index.tolist()
    return bad_words, scores_series

def compute_co_occurrence_matrix(corpus:List[List[str]], window_size=1):
    """Compute co-occurrence matrix for the given corpus with specified window size.
    Args:
        corpus: List of sentences, where each sentence is a list of words.
        window_size: Size of the context window on each side of the target word.
    Returns:
        co_occurrence_matrix: 2D numpy array representing the co-occurrence counts.
        word_list: List of unique words corresponding to the matrix indices.
        word_to_index: Dictionary mapping words to their indices in the matrix.
    """
    all_tokens = list(chain(*corpus))
    word_list = sorted(list(set(all_tokens)))
    word_to_index = {word: i for i, word in enumerate(word_list)}
    vocab_size = len(word_list)

    co_occurrence_counts = defaultdict(int)

    for sentence in corpus:
        indices = [word_to_index[word] for word in sentence]
        
        for i, target_idx in enumerate(indices):
            start_index = max(0, i - window_size)
            end_index = min(len(indices), i + window_size + 1)

            for j in range(start_index, end_index):
                if i == j:
                    co_occurrence_counts[(target_idx, target_idx)] += 1
                    continue
                
                context_idx = indices[j]
                
                pair_key = tuple(sorted((target_idx, context_idx)))
                co_occurrence_counts[pair_key] += 1

    co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int16)

    for (idx_a, idx_b), count in co_occurrence_counts.items():
        co_occurrence_matrix[idx_a, idx_b] = count
        if idx_a != idx_b:
            co_occurrence_matrix[idx_b, idx_a] = count
        
    return co_occurrence_matrix, word_list, word_to_index

def analyser_anisotropie_advanced(embeddings, max_samples=10000):
    """
    Advanced anisotropy analysis with mathematical fixes and PC alignment.
    
    Args:
        embeddings: np.array (n_samples, n_features)
        max_samples: int, limit for pairwise calculation to avoid RAM OOM.
    """
    n, d = embeddings.shape
    
    # ---- 0. Sampling for Pairwise Ops (Speed/RAM Safety) ----
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        sample_emb = embeddings[indices]
    else:
        sample_emb = embeddings

    # ---- 1. Normalization ----
    # We use the sample for expensive stats, full for global stats
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    emb_norm = sample_emb / (norms + 1e-10)
    
    # Full dataset normalization for PCA
    full_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    full_emb_norm = embeddings / (full_norms + 1e-10)

    print(f"--- Geometry Analysis (N={n}, D={d}) ---")

    # ---- 2. Pairwise Cosine Statistics ----
    # High mean + Low Std = The "Narrow Cone" Effect
    sim_matrix = cosine_similarity(emb_norm)
    # Extract upper triangle to avoid self-similarity (1.0) and duplicates
    i, j = np.triu_indices_from(sim_matrix, k=1)
    sims = sim_matrix[i, j]

    print(f"Mean Cosine Sim:      {sims.mean():.4f}  (Target: ~0 for isotropic)")
    print(f"Std Cosine Sim:       {sims.std():.4f}")
    print(f"Distribution Skew:    {skew(sims):.4f}  (>0 means skewed toward alignment)")

    # ---- 3. The Ethayarajh Metric (Vector Space Drift) ----
    # Does the space define a specific direction?
    global_mean_vec = full_emb_norm.mean(axis=0)
    mean_vec_norm = np.linalg.norm(global_mean_vec)
    
    print(f"Norm of Mean Vector:  {mean_vec_norm:.4f}  (0=Isotropic, 1=Collapsed)")

    # ---- 4. PCA & Effective Rank (Dimensionality) ----
    pca = PCA(n_components=min(d, n))
    pca.fit(full_emb_norm)
    evr = pca.explained_variance_ratio_
    
    # Effective Rank (Evans et al., 2022)
    # Measures the "true" number of dimensions being used
    entropy = -np.sum(evr * np.log(evr + 1e-12))
    eff_rank = np.exp(entropy)
    
    print(f"Top-1 Variance:       {evr[0]:.4f}")
    print(f"Effective Rank:       {eff_rank:.1f} / {d}")

    # ---- 5. Alignment to Principal Components (New) ----
    # Are vectors just pointing at the top eigenvector?
    # We project unit vectors onto the first Principal Component
    pc1 = pca.components_[0]
    # Dot product (cosine since vectors are normalized)
    alignment_pc1 = np.abs(full_emb_norm @ pc1)
    print(f"Avg Alignment to PC1: {alignment_pc1.mean():.4f}  (1.0 = perfectly aligned on PC1)")

    # ---- 6. Whitening Impact (Corrected Math) ----
    # We calculate how much 'centering' alone fixes the problem.
    # If Centering fixes it -> The anisotropy is just a shift (easy fix).
    # If Whitening is needed -> The anisotropy is scaling/correlation (harder).
    
    centered = emb_norm - emb_norm.mean(axis=0)
    
    # Calculate mean cosine of centered data
    sim_centered = cosine_similarity(centered)
    sims_c = sim_centered[np.triu_indices_from(sim_centered, k=1)]
    
    print(f"\n--- Remediation Check ---")
    print(f"Original Mean Cosine: {sims.mean():.4f}")
    print(f"Centered Mean Cosine: {sims_c.mean():.4f}  (Lower is better)")
    
    # Quick check: If Centered Mean Cosine is close to 0, you don't strictly need Whitening.
    if sims_c.mean() < 0.05:
        print(">> Result: Centering alone restores isotropy.")
    else:
        print(">> Result: Deep anisotropy detected (requires whitening/flow-based correction).")

    return {
        "mean_cosine": sims.mean(),
        "effective_rank": eff_rank,
        "pc1_alignment": alignment_pc1.mean()
    }
    
def normalize_range_center(intonations:List[List[int]] | List[int], range_normalize:float=1.0, 
                           center:float=1.0) -> List[List[float]]:
    if isinstance(intonations[0], list):
        all_intonations = [inton for sublist in intonations for inton in sublist]
    else :
        all_intonations = intonations
    min_inton = min(all_intonations) # Find the minimum intonation value
    max_inton = max(all_intonations) # Find the maximum intonation value
    assert max_inton > min_inton, "Error: All intonation values are the same."

    normalized_intonations = []
    for sentence in intonations:
        normalized_sentence = [
            (inton - min_inton) / (max_inton - min_inton) for inton in sentence
        ] # Normalize to [0, 1]
        normalized_sentence = [
            range_normalize * intonation + (center - range_normalize / 2)
            for intonation in normalized_sentence
        ]
        normalized_intonations.append(normalized_sentence)

    return normalized_intonations

def make_binary(intonations: List[List[int]], threshold: float = None, zero_value:int=0, one_value:int=0) -> List[List[float]]:
    """
    Convert intonation values to binary representation based on a threshold.
    This function takes a list of intonation values organized by sentences and converts
    each value to either a zero_value or one_value based on comparison with a threshold.
    If no threshold is provided, it defaults to the mean of all intonation values.
    
    Parameters
    ----------
    intonations : List[List[int]]
        A 2D list where each sublist contains intonation values for a sentence.
    threshold : float, optional
        The cutoff value for binary conversion. Values >= threshold
        are converted to one_value, values < threshold are converted to zero_value.
        If None, defaults to the mean of all intonation values. Defaults to None.
    zero_value : int, optional
        The value to assign for intonations below the threshold. Defaults to 0.
    one_value : int, optional
        The value to assign for intonations at or above the threshold. Defaults to 0.
    
    Returns
    -------
    List[List[float]]
        A 2D list with the same structure as intonations, where each
        value has been converted to either zero_value or one_value based on the threshold.
    
    Raises
    -------
    AssertionError: 
        If all intonation values are identical (max_inton == min_inton).
        
    Example:
    -------
    >>> intonations = [[100, 150, 120], [90, 110]]
    >>> make_binary(intonations, zero_value=0, one_value=1)
        [[0, 1, 0], [0, 1]]
    """
    all_intonations = [inton for sublist in intonations for inton in sublist]
    
    if not all_intonations:
        return []
        
    min_inton = min(all_intonations)
    max_inton = max(all_intonations)
    assert max_inton > min_inton, "Error: All intonation values are the same."

    if threshold is None:
        threshold = sum(all_intonations) / len(all_intonations)

    binary_intonations = []
    for sentence in intonations:
        binary_sentence = [one_value if inton >= threshold else zero_value for inton in sentence]
        binary_intonations.append(binary_sentence)

    return binary_intonations

def create_all_clusters(all_cosine_similarity: pd.DataFrame, min_similarity: float = 0.5) -> dict[int, list[str]]:
    """
    Performs hierarchical agglomerative clustering on a similarity matrix.

    Args:
        all_cosine_similarity (pd.DataFrame): The pairwise similarity matrix from `create_cos_similarity`.
        min_similarity (float): The minimum similarity required for words to be grouped together. 
                                A higher value (e.g., 0.8) creates smaller, tighter clusters.

    Returns:
        Dict[int, List[str]]: A dictionary mapping a unique cluster ID (int) to its list of words.
    """
    dist_matrix = 1.0 - all_cosine_similarity.values
    np.fill_diagonal(dist_matrix, 0.0) 
    
    condensed_dist = ssd.squareform(dist_matrix)
    Z = linkage(condensed_dist, method='complete')
    distance_threshold = 1.0 - min_similarity
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
    
    clusters: dict[int, list[str]] = {}
    words = all_cosine_similarity.index
    
    for word, label in zip(words, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)
        
    return clusters
    
def find_good_clusters(clusters: dict[int, list[str]], df_similarity: pd.DataFrame, min_words: int) -> list[tuple[int, float, list[str]]]:
    """
    Filters clusters by size and calculates the average internal cohesion (tightness) of each.

    Args:
        clusters (Dict[int, List[str]]): The output from `create_all_clusters`.
        df_similarity (pd.DataFrame): The original similarity matrix.
        min_words (int): The minimum number of words a cluster must contain to be kept.

    Returns:
        List[Tuple[int, float, List[str]]]: A list of valid clusters sorted by cohesion (highest first).
                                            Format: [(cluster_id, avg_similarity, word_list), ...]
    """
    valid_clusters = []
    
    for cluster_id, words in clusters.items():
        if len(words) >= min_words:
            sub_matrix = df_similarity.loc[words, words]
            mask = np.triu(np.ones(sub_matrix.shape), k=1).astype(bool)
            
            avg_similarity = sub_matrix.values[mask].mean()
            std_similarity = sub_matrix.values[mask].std()
            
            valid_clusters.append((cluster_id, avg_similarity, std_similarity, words))
            
    valid_clusters.sort(key=lambda x: x[1], reverse=True)
            
    return valid_clusters

def compute_all_sim(vecs:np.ndarray, words:list[str])-> pd.DataFrame:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normalized:np.ndarray = vecs / (norms + 1e-10)
    similarity_matrix = np.dot(vecs_normalized, vecs_normalized.T)
    return pd.DataFrame(similarity_matrix, columns=words, index=words)

