from typing import List
import torch
from torch.utils.data import Dataset

from collections import Counter

import random

class SGNS_store_DataSet(Dataset):
    """ 
    Dataset pour word2vec avec les négatives sampling.
    """
    def _dist_unigram(self, power:float):
        freq_list = [self.freq.get(self.decoder[i], 0) for i in range(self.vocab_size)]
        unigram = torch.tensor([f**power for f in freq_list], dtype=torch.float)
        return unigram / unigram.sum()
    
    def _proba_words(self):
        if self.subsample_thresh > 0.0:
            total_tokens = sum(self.freq.values())
            word_probs = {}
            t = self.subsample_thresh * total_tokens
            for w, c in self.freq.items():
                f = c
                prob_keep = ((f / t)**0.5 + 1) * (t / f)
                word_probs[w] = min(1.0, prob_keep)
        else:
            word_probs = None
        return word_probs

    def _make_pairs(self):
        self.pairs = []
        for sent in self.sentences:
            if self.word_probs is not None:
                filtered = [w for w in sent if random.random() < self.word_probs.get(w, 1.0)]
            else:
                filtered = sent
            ids = [self.encoder[w] for w in filtered if w in self.encoder]
            L = len(ids)
            for i, center in enumerate(ids):
                cur_window = self.context_size
                start = max(0, i - cur_window)
                end = min(L, i + cur_window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = ids[j]
                    self.pairs.append((center, context))

    def __init__(self, sentences:list[list[str]], window_size:int=2, nb_neg:int=5, power=0.75,
                 subsample_thresh:float=1e-5 , vocab_freq:None|dict|Counter=None, vocab_size_limit:None|int=None):
        """Initialise le dataset pour du Word2Vec avec des pairs négative. (Warning Méthode qui stocke en mémoire)
        
        Args:
            sentences: Liste des phrases du corpus de texte (une phrases doit être une liste de str)
            window_size: La taille de fenêtre pour créer les pairs positif.
            nb_neg: Nombre de pair négatif pour chaque mots. (K)
            subsample_thresh: Pour réduire la fréquence des mots trop fréquent (Ex : de, le, la, ...) dans le choix des mots centraux
            power: Pour réduire la fréquence des mots trop fréquent dans les négatifs
            vocab_freq: Dictionnaire ou counter (https://docs.python.org/3/library/collections.html#counter-objects) pour chaque mots indique la fréquence de se mot dans tout le corpus
            vocab_size_limit: Pour ne garder que les top-N mots par fréquence
        """
        super().__init__()
        subsample_thresh = float(subsample_thresh)
        # assert isinstance(sentences, list[list[str]]), "sentences should be a list[list[str]]"
        assert isinstance(window_size, int), "window_size should be a int"
        assert isinstance(nb_neg, int), "nb_neg should be a int"
        assert isinstance(subsample_thresh, float), "subsample_thresh should be a float"

        self.sentences:list[list[str]] = sentences
        self.context_size:int = window_size
        self.K:int = nb_neg
        self.power = power

        if vocab_freq is not None:
            full_freq:Counter = Counter(vocab_freq)
        else:
            all_tokens = [t for s in sentences for t in s]
            full_freq:Counter = Counter(all_tokens)

        if vocab_size_limit is not None:
            most_common = full_freq.most_common(vocab_size_limit)
            kept_words = [w for w, _ in most_common]
            # On recalcul la fréquence des mots
            self.freq = Counter({w: full_freq[w] for w in kept_words})
        else:
            self.freq = full_freq

        self.vocab = sorted(list(self.freq.keys()))
        self.vocab_size:int = len(self.vocab)
        self.encoder:dict = {w:i for i,w in enumerate(self.vocab)}
        self.decoder = {i:w for w,i in self.encoder.items()}

        self.unigram_dist = self._dist_unigram(self.power)

        self.subsample_thresh:float = subsample_thresh
        self.word_probs:dict = self._proba_words()
        self.pairs = None
        self._make_pairs()
        assert len(self.pairs) != 0 , "Error to make positif pairs"

    def __len__(self):
        return len(self.pairs)
    
    def _sample_negatives(self, batch_size):
        """
        Échantillonne (batch_size, K) négatifs selon self.unigram_dist.
        
        Args:
            batch_size: La taille de batch [B]

        Return:
           torch.LongTensor
        """
        # torch.multinomial attend vecteur de probabilités ; on échantillonne batch_size*K et reshape
        neg = torch.multinomial(self.unigram_dist, batch_size * self.K, replacement=True)
        return neg.view(batch_size, self.K)    

    def __getitem__(self, idx):
        """Prends la pairs positif idx (idx >= 0 and idx < len(self.pairs) et les négatifs qui sont calculé avec la distribution unigram

        Args:
            idx: Index de la pairs positifs.

        Return :
            Tuple(center_id:Long, pos_id:long, negatives_ids: torch.LongTensor de taille [K])
        - center_id: long
        - pos_id: long
        - negatives: torch.LongTensor shape [K]
        """
        center, pos = self.pairs[idx]
        neg = torch.multinomial(self.unigram_dist, self.K, replacement=True)
        return torch.tensor(center, dtype=torch.long), torch.tensor(pos, dtype=torch.long), neg

    def collate_batch(self, batch):
        """Fonction de collate pour DataLoader.
        Args:
            batch: list of tuples (center, pos, neg) où neg is tensor [K]
        Return:
            Tuple(centers:Torch.Tensor [B], pos:Torch.Tensor [B], neg:Torch.Tensor [B, K])
        """
        centers = torch.stack([item[0] for item in batch], dim=0)
        pos = torch.stack([item[1] for item in batch], dim=0)
        negs = torch.stack([item[2] for item in batch], dim=0)
        return centers, pos, negs

    def sample_batch_negatives(self, centers, K=None):
        """Méthode pour voir un échantillons de pairs négatifs sur un batch de mots centraux
        Args:
            centers: Tensor d'idx des mots centraux
        Return:
            Torch.Tensor() [B K]
        """
        B = centers.size(0)
        K = self.K if K is None else K
        return torch.multinomial(self.unigram_dist, B * K, replacement=True).view(B, K)
    
    def encode(self, words:list|str) -> list|int:
        if isinstance(words, str) : return self.encoder[words]
        ids = []
        for w in words :
            ids.append(self.encoder[w])
        return ids
    
    def decode(self, ids:list|int) -> list|int:
        if isinstance(ids, int) : return self.decoder[ids]
        words = []
        for i in ids :
            words.append(self.decoder[i])
        return words

class W2V_weighted_DataSet(Dataset):
    def compute_importance(self, words, intonations):
        dict_list_importance = {}
        for sentence, intonation in zip(words, intonations) :
            for index, inton in enumerate(intonation):
                if sentence[index] not in dict_list_importance :
                    dict_list_importance[sentence[index]] = [float(inton)]
                else :
                    dict_list_importance[sentence[index]].append(float(inton))

        dict_importance = {}
        for word in dict_list_importance :
            dict_importance[word] = sum(dict_list_importance[word]) / len(dict_list_importance[word])

        return dict_importance
    
    def _get_unigram_dist(self):
        """Compute unigram distribution depending on word importance"""
        weight_list = [self.word_importance[token] for token in range(len(self.encoder)) ]
        unigram = torch.tensor([weight for weight in weight_list], dtype=torch.float)
        return unigram / unigram.sum()

    
    def _make_pairs_positif(self):
        pairs = []
        for sent, intonation in zip(self.sentences, self.intonations):
            ids = self.encode(sent)
            L = len(ids)
            for i, center in enumerate(ids):
                cur_window = self.context_size
                start = max(0, i - cur_window)
                end = min(L, i + cur_window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = ids[j]
                    pairs.append((center, context, intonation[i]))
        return pairs
    
    def __init__(self, sentences:list[list[str]], intonations:List[List[float]] , window_size:int=2, nb_neg:int=5):
        super().__init__()
        
        assert len(sentences) == len(intonations), f"Error: Sentences and intonations must have the same length."

        all_tokens = [t for sentence in sentences for t in sentence if t.isalpha()]
        self.vocab = list(set(all_tokens))
        self.encoder:dict = {w:i for i,w in enumerate(self.vocab)}
        self.decoder:dict = {i:w for i,w in enumerate(self.vocab)}
        self.context_size:int = window_size
        self.sentences = sentences
        self.intonations = intonations
        self.K = nb_neg
        
        self.tokens = []
        for s in sentences :
            self.tokens.append([])
            for w in s :
                self.tokens[-1] .append(self.encoder[w])

        self.word_importance:dict = self.compute_importance(self.tokens, intonations)
        self.unigram_dist = self._get_unigram_dist()
        self.pairs:List = self._make_pairs_positif()

    def encode(self, words:list|str) -> list|int:
        if isinstance(words, str) : return self.encoder[words]
        ids = []
        for w in words :
            ids.append(self.encoder[w])
        return ids
    
    def decode(self, ids:list|int) -> list|int:
        if isinstance(ids, int) : return self.decoder[ids]
        words = []
        for i in ids :
            words.append(self.decoder[i])
        return words

    def __getitem__(self, idx:int):
        center, pos, intonation = self.pairs[idx]
        neg = torch.multinomial(self.unigram_dist, self.K, replacement=True)
        return center, pos, neg, intonation
    
    def __len__(self):
        return len(self.pairs)
