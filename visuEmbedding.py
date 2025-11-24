import torch
from torch.nn import Embedding
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from typing import Sequence, Optional, List, Any

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from copy import deepcopy

def interactive_embedding_plot_3D(embedding: Embedding,
                               encoder: dict,
                               decoder:dict,
                               words: Optional[Sequence[str]] = None,
                               method: Optional[str] = "pca",            # "pca" or "tsne" or None
                               n_components: int = 3,
                               tsne_params: dict = None,
                               top_k_neighbors: int = 10,
                               query_word: Optional[List[str]|str] = None,
                               show_only: Optional[Sequence[str]] = None,
                               title: Optional[str] = None) -> go.Figure :
    """
    Trace un scatter interactif (Plotly) des embeddings.
    - embedding: nn.Embedding (poids seront copiés en cpu)
    - encoder: dict word -> idx
    - words: séquence de mots à afficher (par défaut tous les mots de encoder)
    - method: 'pca' ou 'tsne'
    - tsne_params: dict de paramètres passés à TSNE si method == 'tsne'
    - top_k_neighbors: nombre de voisins affichés si query_word spécifié
    - query_word: si fourni, on le met en évidence et on affiche ses k voisins les plus proches
    - show_only: si fourni, restreint l'affichage à cette liste de mots
    - title: titre du graphique
    Retour: Plotly Figure (affiche inline si en notebook)
    """
    # Préparer la liste de mots à afficher
    if words is None:
        words = list(encoder.keys())
    words = list(words)

    if show_only is not None:
        show_set = set(show_only)
        words = [w for w in words if w in show_set]

    # Construire la matrice d'embeddings [N, D]
    idxs = [encoder[w] for w in words]
    with torch.no_grad():
        W = embedding.weight.cpu().numpy()[idxs]  # shape [N, D]

    # Réduction de dimension
    if method is None:
        X = W
    elif method.lower() == "pca":
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(W)
    elif method.lower() == "tsne":
        tsne_params = tsne_params or {"perplexity": 30, "n_iter": 1000, "init": "pca", "learning_rate": 200}
        tsne = TSNE(n_components=n_components, **tsne_params)
        X = tsne.fit_transform(W)
    else:
        raise ValueError("method must be 'pca' or 'tsne' or None")

    # Construire dataframe-like arrays
    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    labels = words

    # Préparer hover text (word + optional nearest neighbors cos sim quick)
    hover_text = []
 
    for w in labels:
        hover_text.append(w)

    fig:go.Figure = px.scatter_3d(x=xs, y=ys, z=zs, hover_name=labels, title=(title or "Embeddings 3D"),
                    width=1000, height=800)
    
    # add invisible text markers for readability; show points as small markers
    fig.update_traces(marker=dict(size=4, opacity=0.8))

    # If query_word provided, highlight it and its neighbors
    if query_word is not None:
        if query_word not in encoder:
            raise KeyError(f"query_word '{query_word}' not found in encoder")
        # compute cosine similarities between query embedding and all displayed embeddings
        with torch.no_grad():
            query_idx = encoder[query_word]
            query_vec = embedding.weight.cpu().numpy()[query_idx:query_idx+1]  # [1, D]
            sims = cosine_similarity(query_vec, W).flatten()  # [N]
            # get top_k indices (exclude itself if present)
            order = np.argsort(-sims)
            # keep top_k_neighbors (including query if in list)
            topk = order[:top_k_neighbors]
        # build lines to neighbors and highlight markers
        neighbor_words = [labels[i] for i in topk]
        neighbor_sims = sims[topk]

        # highlight query point (if it's in displayed words)
        if query_word in labels:
            qpos = labels.index(query_word)
            fig.add_trace(go.Scatter3d(x=[xs[qpos]], y=[ys[qpos]], z=[zs[qpos]],
                           mode='markers+text',
                           marker=dict(size=8, color='red', symbol='diamond'),
                           text=[query_word], textposition='top center', name=f'query {query_word}'))
        # draw neighbors and lines
        for i, ni in enumerate(topk):
            if labels[ni] == query_word:
                continue
            fig.add_trace(go.Scatter3d(x=[xs[ni]], y=[ys[ni]], z=[zs[ni]],
                           mode='markers+text',
                           marker=dict(size=6, color='orange'),
                           text=[labels[ni]], textposition='top center',
                           name=f'neighbor_{i} (sim={neighbor_sims[i]:.3f}) {decoder[ni]}'))
            
            # line from query to neighbor if query is displayed
            if query_word in labels:
                fig.add_trace(go.Scatter3d(x=[xs[qpos], xs[ni]], y=[ys[qpos], ys[ni]], z=[zs[qpos], zs[ni]],
                           mode='lines', line=dict(width=2, color='gray'), showlegend=False))
    

    max_range = np.max(np.ptp(X, axis=0))  # plage maximale sur les composants projetés
    if max_range == 0:
        max_range = 1.0
    scale = max_range * 0.05  # longueur des axes (ajuste si besoin)

    # axes X, Y, Z : lignes de -scale à +scale
    axes_traces = [
        # X axis (rouge)
        go.Scatter3d(x=[-scale, scale], y=[0, 0], z=[0, 0],
                     mode='lines', line=dict(color='red', width=4), name='axis X'),
        # Y axis (vert)
        go.Scatter3d(x=[0, 0], y=[-scale, scale], z=[0, 0],
                     mode='lines', line=dict(color='green', width=4), name='axis Y'),
        # Z axis (bleu)
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[-scale, scale],
                     mode='lines', line=dict(color='blue', width=4), name='axis Z'),
    ]

    # flèches aux extrémités (petits segments pour simuler des flèches) et labels
    arrow_len = scale * 0.008
    arrow_traces = []
    labels_traces = []
    # X positive arrow
    arrow_traces.append(go.Scatter3d(x=[scale, scale - arrow_len], y=[0, 0], z=[0, 0],
                                     mode='lines', line=dict(color='red', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[scale], y=[0], z=[0], mode='text',
                                      text=['X'], textposition='top center', showlegend=False))
    # Y positive arrow
    arrow_traces.append(go.Scatter3d(x=[0, 0], y=[scale, scale - arrow_len], z=[0, 0],
                                     mode='lines', line=dict(color='green', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[0], y=[scale], z=[0], mode='text',
                                      text=['Y'], textposition='top center', showlegend=False))
    # Z positive arrow
    arrow_traces.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[scale, scale - arrow_len],
                                     mode='lines', line=dict(color='blue', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[0], y=[0], z=[scale], mode='text',
                                      text=['Z'], textposition='top center', showlegend=False))

    # ajouter toutes les traces au fig
    for t in axes_traces + arrow_traces + labels_traces:
        fig.add_trace(t)


    fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
    return fig

def components_to_fig_3D(components, encoder:dict, highlight_words:Optional[List]=None, words_display:Optional[List]=None,
                        nb_neighbors:int=2, title:str="figure of vector", _min:float=-5, _max:float=5, base_color:dict={}) -> go.Figure :
    """
    """
    if words_display is None:
        words_display = list(encoder.keys())

    # Construire la matrice d'embeddings [N, D]
    idxs = [encoder[w] for w in words_display]

    # Keep only components of words_display
    components_wd = components[idxs]
    xs = components_wd[:,0]
    ys = components_wd[:,1]
    zs = components_wd[:,2]

    fig:go.Figure = px.scatter_3d(x=xs, y=ys, z=zs, hover_name=words_display, title=title,
                    width=1000, height=800)
    
    # add invisible text markers for readability; show points as small markers
    fig.update_traces(marker=dict(size=4, opacity=0.8))

    # If query_word provided, highlight it and its neighbors
    pos_map = {w:i for i, w in enumerate(words_display)}

    if highlight_words is not None:
        # compute cosine similarities between query embedding and all displayed embeddings
        for w in highlight_words:
            if w in base_color:
                color_word, color_neighbor = base_color[w]
            else :
                color_word, color_neighbor = ('red', 'orange')
            query_idx = encoder[w]
            query_vec = components[query_idx:query_idx+1]  # [1, D]
            sims = cosine_similarity(query_vec, components_wd).flatten()  # [N]
            # get top_k indices (exclude itself if present)
            order = np.argsort(-sims)
            # keep top_k_neighbors (including query if in list)
            topk = order[:nb_neighbors]
            # build lines to neighbors and highlight markers
            # neighbor_words = [words_display[i] for i in topk]
            neighbor_sims = sims[topk]

            # highlight query point (if it's in displayed words)
            qpos = pos_map.get(w, None)
            if qpos is None:
                continue
            fig.add_trace(go.Scatter3d(x=[xs[qpos]], y=[ys[qpos]], z=[zs[qpos]],
                        mode='markers+text',
                        marker=dict(size=8, color=color_word, symbol='diamond'),
                        text=[w], textposition='top center', name=f'query {w}'))
            
            # draw neighbors and lines
            for i, ni in enumerate(topk):
                if words_display[ni] == w:
                    continue
                fig.add_trace(go.Scatter3d(x=[xs[ni]], y=[ys[ni]], z=[zs[ni]],
                            mode='markers+text',
                            marker=dict(size=6, color=color_neighbor),
                            text=[words_display[ni]], textposition='top center',
                            name=f'neighbor_{i} (sim={neighbor_sims[i]:.3f}) {words_display[ni]}'))
                
                # line from query to neighbor if query is displayed
                if w in words_display:
                    fig.add_trace(go.Scatter3d(x=[xs[qpos], xs[ni]], y=[ys[qpos], ys[ni]], z=[zs[qpos], zs[ni]],
                            mode='lines', line=dict(width=2, color='gray'), showlegend=False))
    

    max_range = np.max(np.ptp(components_wd, axis=0))
    if max_range == 0:
        max_range = 1.0
    scale = max_range * 0.05

    axes_traces = [
        go.Scatter3d(x=[-scale, scale], y=[0, 0], z=[0, 0],
                     mode='lines', line=dict(color='red', width=4), name='axis X'),
        go.Scatter3d(x=[0, 0], y=[-scale, scale], z=[0, 0],
                     mode='lines', line=dict(color='green', width=4), name='axis Y'),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[-scale, scale],
                     mode='lines', line=dict(color='blue', width=4), name='axis Z'),
    ]

    arrow_len = scale * 0.008
    arrow_traces = []
    labels_traces = []
    arrow_traces.append(go.Scatter3d(x=[scale, scale - arrow_len], y=[0, 0], z=[0, 0],
                                     mode='lines', line=dict(color='red', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[scale], y=[0], z=[0], mode='text',
                                      text=['X'], textposition='top center', showlegend=False))
    arrow_traces.append(go.Scatter3d(x=[0, 0], y=[scale, scale - arrow_len], z=[0, 0],
                                     mode='lines', line=dict(color='green', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[0], y=[scale], z=[0], mode='text',
                                      text=['Y'], textposition='top center', showlegend=False))
    arrow_traces.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[scale, scale - arrow_len],
                                     mode='lines', line=dict(color='blue', width=6), showlegend=False))
    labels_traces.append(go.Scatter3d(x=[0], y=[0], z=[scale], mode='text',
                                      text=['Z'], textposition='top center', showlegend=False))

    for t in axes_traces + arrow_traces + labels_traces:
        fig.add_trace(t)


    fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[_min, _max], title='PC1'),
            yaxis=dict(nticks=4, range=[_min, _max], title='PC2'),
            zaxis=dict(nticks=4, range=[_min, _max], title='PC3')
        ), scene_aspectmode='cube')
    return fig


def components_to_fig_3D_animation(history_components, encoder:dict, highlight_words:Optional[List]=None, 
                                words_display:Optional[List]=None, nb_neighbors:int=2, title:str="figure of vector",
                                _min:float=-5, _max:float=5, base_color:dict={}) -> go.Figure :
    """
    Crée une animation 3D de l'évolution de l'embedding.
    Utilise ta fonction existante components_to_fig_3D pour chaque frame.

    ----------
    INPUT :
    ----------
    history_components : list[np.ndarray]
        Liste d'embeddings successifs (N,3)
    encoder : dict
    words_display : list[str] or None
    highlight_words : list[str] or None
    nb_neighbors : int
    title : str

    ----------
    OUTPUT :
    ----------
    fig : go.Figure
        Animation 3D Plotly
    """
    frames = []
    for step, comp in enumerate(history_components):
        base_fig = components_to_fig_3D(
            components=comp,
            encoder=encoder,
            highlight_words=highlight_words,
            words_display=words_display,
            nb_neighbors=nb_neighbors,
            title=f"{title} — step {step}",
            _min=_min, _max=_max, base_color=base_color
        )
        frame_traces = deepcopy(base_fig.data)

        frames.append(
            go.Frame(data=frame_traces, name=f"step_{step}")
        )

    init_fig = components_to_fig_3D(
        components=history_components[-1],
        encoder=encoder,
        highlight_words=highlight_words,
        words_display=words_display,
        nb_neighbors=nb_neighbors,
        title=title, base_color=base_color,
        _min=_min, _max=_max
    )

    fig = go.Figure(
        data=deepcopy(init_fig.data),
        layout=init_fig.layout,
        frames=frames
    )

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}}]}
                ],
                "x": 0.1, "y": 1.1,
                "showactive": False
            }
        ],
        sliders=[{
            "pad": {"b": 10},
            "steps": [
                {"label": f"{i}",
                 "method": "animate",
                 "args": [[f"step_{i}"]]}
                for i in range(len(history_components))
            ]
        }]
    )

    fig.show()

    return fig   

def interactive_embedding_plot_2D(embedding: torch.nn.Embedding,
                               encoder: dict,
                               decoder:dict,
                               words: Optional[Sequence[str]] = None,
                               method: Optional[str] = "pca",            # "pca" or "tsne" or None
                               n_components: int = 2,
                               tsne_params: dict = None,
                               top_k_neighbors: int = 10,
                               query_word: Optional[List[str]|str] = None,
                               show_only: Optional[Sequence[str]] = None,
                               title: Optional[str] = None):
    """
    Trace un scatter interactif (Plotly) des embeddings.
    - embedding: nn.Embedding (poids seront copiés en cpu)
    - encoder: dict word -> idx
    - words: séquence de mots à afficher (par défaut tous les mots de encoder)
    - method: 'pca' ou 'tsne'
    - tsne_params: dict de paramètres passés à TSNE si method == 'tsne'
    - top_k_neighbors: nombre de voisins affichés si query_word spécifié
    - query_word: si fourni, on le met en évidence et on affiche ses k voisins les plus proches
    - show_only: si fourni, restreint l'affichage à cette liste de mots
    - title: titre du graphique
    Retour: Plotly Figure (affiche inline si en notebook)
    """
    # Préparer la liste de mots à afficher
    if words is None:
        words = list(encoder.keys())
    words = list(words)

    if show_only is not None:
        show_set = set(show_only)
        words = [w for w in words if w in show_set]

    # Construire la matrice d'embeddings [N, D]
    idxs = [encoder[w] for w in words]
    device = embedding.weight.device
    with torch.no_grad():
        W = embedding.weight.cpu().numpy()[idxs]  # shape [N, D]

    # Réduction de dimension
    if method.lower() == "pca":
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(W)
    elif method.lower() == "tsne":
        tsne_params = tsne_params or {"perplexity": 30, "n_iter": 1000, "init": "pca", "learning_rate": 200}
        tsne = TSNE(n_components=n_components, **tsne_params)
        X = tsne.fit_transform(W)
    elif method is None:
        X = W[0:n_components]
    else:
        raise ValueError("method must be 'pca' or 'tsne' or None")

    # Construire dataframe-like arrays
    xs = X[:, 0]
    ys = X[:, 1] if n_components >= 2 else np.zeros(len(xs))
    labels = words

    # Préparer hover text (word + optional nearest neighbors cos sim quick)
    hover_text = []
    # compute cosine similarities matrix once if needed for hover top neighbors
    if top_k_neighbors and query_word is not None:
        # we'll compute neighbors for query only below
        cos_sim_matrix = None
    else:
        cos_sim_matrix = None

    for w in labels:
        hover_text.append(w)

    fig = px.scatter(x=xs, y=ys, text=["" for _ in labels], hover_name=labels,
                     title=(title or "Embeddings"), width=900, height=700)

    # add invisible text markers for readability; show points as small markers
    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))

    # If query_word provided, highlight it and its neighbors
    if query_word is not None:
        if query_word not in encoder:
            raise KeyError(f"query_word '{query_word}' not found in encoder")
        # compute cosine similarities between query embedding and all displayed embeddings
        with torch.no_grad():
            query_idx = encoder[query_word]
            query_vec = embedding.weight.cpu().numpy()[query_idx:query_idx+1]  # [1, D]
            sims = cosine_similarity(query_vec, W).flatten()  # [N]
            # get top_k indices (exclude itself if present)
            order = np.argsort(-sims)
            # keep top_k_neighbors (including query if in list)
            topk = order[:top_k_neighbors]
        # build lines to neighbors and highlight markers
        neighbor_words = [labels[i] for i in topk]
        neighbor_sims = sims[topk]

        # highlight query point (if it's in displayed words)
        if query_word in labels:
            qpos = labels.index(query_word)
            fig.add_trace(go.Scatter(x=[xs[qpos]], y=[ys[qpos]],
                                     mode='markers+text',
                                     marker=dict(size=12, color='red', symbol='diamond'),
                                     text=[query_word],
                                     textposition='top center',
                                     hoverinfo='skip',
                                     name=f'query {query_word}'))
        # draw neighbors and lines
        for i, ni in enumerate(topk):
            if labels[ni] == query_word:
                continue
            fig.add_trace(go.Scatter(x=[xs[ni]], y=[ys[ni]],
                                     mode='markers+text',
                                     marker=dict(size=9, color='orange'),
                                     text=[labels[ni]],
                                     textposition='top center',
                                     name=f'neighbor_{i} (sim={neighbor_sims[i]:.3f}) {decoder[ni]}'))
            # line from query to neighbor if query is displayed
            if query_word in labels:
                fig.add_trace(go.Scatter(x=[xs[qpos], xs[ni]], y=[ys[qpos], ys[ni]],
                                         mode='lines',
                                         line=dict(width=1, color='grey'),
                                         hoverinfo='none',
                                         showlegend=False))

    fig.update_layout(clickmode='event+select')
    fig.show()
    return fig

def plot_similarity_heatmap(embedding: torch.nn.Embedding, encoder: dict, words: list[str], figsize=(10, 8)):
    """
    Produit une heatmap de similarité cosinus pour un ensemble de mots.
    
    Args:
        embedding: nn.Embedding contenant les poids des mots.
        encoder: dictionnaire mapping words -> indices.
        words: liste de mots pour lesquels calculer la heatmap.
        figsize: taille de la figure pour la heatmap (width, height).
    """
    # Vérifier que tous les mots sont dans l'encodeur
    for word in words:
        if word not in encoder:
            raise KeyError(f"Word '{word}' not found in encoder.")

    # Calculer la matrice de similarité
    num_words = len(words)
    similarity_matrix = np.zeros((num_words, num_words))

    for i in range(num_words):
        for j in range(num_words):
            idx1 = torch.tensor([encoder[words[i]]], dtype=torch.long)
            idx2 = torch.tensor([encoder[words[j]]], dtype=torch.long)
            with torch.no_grad():
                v1 = embedding(idx1).squeeze(0)  # [D]
                v2 = embedding(idx2).squeeze(0)  # [D]
                similarity_matrix[i, j] = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    # Créer la heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="magma",
                xticklabels=words, yticklabels=words, cbar=True,
                square=False, linewidths=0.)
    
    plt.title("Heatmap de Similarité Cosinus")
    plt.xlabel("Mots", fontstyle="italic")
    plt.ylabel("Mots", fontstyle="italic")
    plt.savefig("tmp.png")
    plt.show()

def cluster_words(embedding: torch.nn.Embedding, encoder: dict, words: list[str], n_clusters: int = 5, plot: bool = True):
    """
    Effectue des clusters de mots et affiche les résultats en 3D (Plotly).
    Retour: dict: mots groupés par leurs clusters.
    """
    for word in words:
        if word not in encoder:
            raise KeyError(f"Word '{word}' not found in encoder.")

    # Récupérer les embeddings pour les mots (en CPU numpy)
    idxs = [encoder[w] for w in words]
    with torch.no_grad():
        W = embedding.weight.cpu().numpy()[idxs]  # shape [N, D]

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(W)
    labels = kmeans.labels_

    print("labels :")
    print(labels)

    clusters = {i: [] for i in range(n_clusters)}
    for word, cid in zip(words, labels):
        clusters[cid].append(word)

    print("Clusters de mots :")
    for cluster_id, cluster_words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")

    if plot:
        # PCA 3D pour visualisation
        pca = PCA(n_components=3)
        X3 = pca.fit_transform(W)  # [N, 3]

        df_x = X3[:, 0]
        df_y = X3[:, 1]
        df_z = X3[:, 2]

        # couleurs par cluster
        fig = px.scatter_3d(x=df_x, y=df_y, z=df_z,
                            color=[str(l) for l in labels],
                            hover_name=words,
                            title=f'Clustering des mots (KMeans k={n_clusters})',
                            width=1000, height=800)

        # marqueurs
        fig.update_traces(marker=dict(size=5, opacity=0.8))

        # Ajouter centroïdes projetés (optionnel)
        centroids = kmeans.cluster_centers_
        centroids_proj = pca.transform(centroids)
        fig.add_trace(go.Scatter3d(x=centroids_proj[:, 0], y=centroids_proj[:, 1], z=centroids_proj[:, 2],
                                   mode='markers+text',
                                   marker=dict(size=8, color='black', symbol='x'),
                                   text=[f'c{ i }' for i in range(centroids_proj.shape[0])],
                                   textposition='top center',
                                   name='centroids'))

        # Optionnel : annoter quelques points (décommenter si nécessaire)
        # for i, w in enumerate(words):
        #     fig.add_trace(go.Scatter3d(x=[df_x[i]], y=[df_y[i]], z=[df_z[i]],
        #                                mode='text', text=[w], textposition='top center', showlegend=False))

        # Ajouter repère orthonormé centré sur l'origine projetée (si tu veux)
        max_range = np.max(np.ptp(X3, axis=0))
        if max_range == 0:
            max_range = 1.0
        scale = max_range * 0.05
        arrow_len = scale * 0.08
        # axes
        axes = [
            go.Scatter3d(x=[-scale, scale], y=[0, 0], z=[0, 0], mode='lines',
                         line=dict(color='red', width=3), name='X axis'),
            go.Scatter3d(x=[0, 0], y=[-scale, scale], z=[0, 0], mode='lines',
                         line=dict(color='green', width=3), name='Y axis'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[-scale, scale], mode='lines',
                         line=dict(color='blue', width=3), name='Z axis'),
        ]
        for a in axes:
            fig.add_trace(a)

        fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
        fig.show()

    return clusters