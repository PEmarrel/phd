import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


class SkipGramModel(nn.Module):
    def __init__(self, emb_size:int, embedding_dimension:int=15, context_dimension:int|None=None, init_range:float|None=None, sparse:bool=True, device="cpu"):
        """Initialisation du modèle SkipGram
        Args:
            emb_size: La taille de l'embedding, ce nombre devrais être déterminé après le process sur les data, et dépend de la taille de la fenêtre glissante.
            embedding_dimension: La taille souhaité de l'embedding. Pour notre cas d'utilisation nous préférons une taille très petit
            context_dimension: Il n'est pas recommandé de mettre un entier mais de laisser a None.
        
        """
        super().__init__()
        self.emb_size:int = emb_size
        self.emb_dim:int = embedding_dimension
        # On définit pour chaque mots un embedding (soit un vecteur qui représente le mots)
        self.word_emb:nn.Embedding = nn.Embedding(num_embeddings=self.emb_size, embedding_dim=self.emb_dim, device=device, sparse=sparse)

        # Ce deuxième embedding correspond au mots utilisé dans un contexte (!= d'être utiliser comme mot centrale)
        self.con_size = embedding_dimension if context_dimension is None else context_dimension
        self.con_emb:nn.Embedding = nn.Embedding(num_embeddings=self.emb_size, embedding_dim=self.con_size, device=device,sparse=sparse)

        if init_range is None:
            init_range = 0.5 / self.emb_dim
        self.word_emb.weight.data.uniform_(-init_range, init_range)
        self.con_emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, centrals_words:list|torch.Tensor, pos_context:list|torch.Tensor, neg_context:list|torch.Tensor):
        """Fonction du forward pour le modèle SkipGramModel
        Args:
            centrals_words: Liste des ids des tokens des mots centraux [B]
            pos_context: Liste des ids des tokens des mots dans le contexte [B]
            neg_context: Liste des ids des tokens des mots non présent dans le contexte [B, K]
        """
        # B : batch size
        # D : dimension de l'embedding
        # K : Nombre de mots négatifs

        # Pour chaque pair positif, on récupère :
        # Le vecteur du mots centrale (les valeurs de l'embeddding pour le token)
        words_emb:torch.Tensor = self.word_emb(centrals_words) # [B, D]
        # Le vecteur du mots contexte
        context_emb:torch.Tensor = self.con_emb(pos_context) # [B, D]
        # Et les vecteurs des mots négatifs
        neg_emb:torch.Tensor = self.con_emb(neg_context) # [B, K, D]

        # Pour chaque pair on calcul le score de similarité (mots central et contexte positif)
        # positive score: log sigma(u . v_pos)
        pos_score = torch.sum(words_emb * context_emb, dim=1)
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html#logsigmoid pour la LogSigmoid 
        pos_loss = F.logsigmoid(pos_score)

        # On calcul aussi le score de dissimilarité (mots centrals et les mots non présent dans le context)
        # negative score: sum log sigma(-u . v_neg)
        # neg_emb : [B, K, D], on veut multiplier les vecteurs pour chaque mots
        # Il faut donc ajouter une dimension à words_emb (voir https://docs.pytorch.org/docs/stable/generated/torch.bmm.html#torch-bmm)
        # words_emb.unsqueeze(2) : [B, D, 1]
        # See https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch-unsqueeze pour l'ajout de dimension
        neg_score = torch.bmm(neg_emb, words_emb.unsqueeze(-1)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = - (pos_loss + neg_loss).mean()
        return loss
    
    def save_weight(self, path:str="SGNS_weights/"):
        """Sauvegarde des poids des deux embeddings (word embedding et context embedding) dans le dossier path
        Args :
            path: Le dossier dans lequel sauvegarder les poids des deux embeddings
        """
        word_weights = self.word_emb.weight.detach().cpu()
        con_weight = self.con_emb.weight.detach().cpu()
        torch.save(word_weights, path+'word_embedding.pt')
        torch.save(con_weight, path+'con_embedding.pt')

    def load_weight(self, path:str="SGNS_weights/", name_word_weights:str="word_embedding.pt", name_con_weights:str="con_embedding.pt"):
        """Charge les poids depuis un fichier de sauvegarde de pytorch
        Args :
            path: Le dossier où se trouve les deux fichiers des poids
            name_word_weights: Le nom du fichier contenant les poids du word embedding
            name_con_weights: Le nom du fichier contenant les poids du contexte embedding
        """
        word_weights = torch.load(path + name_word_weights)
        con_weight = torch.load(path + name_con_weights)

        self.word_emb:nn.Embedding = nn.Embedding.from_pretrained(word_weights)
        self.con_emb:nn.Embedding = nn.Embedding.from_pretrained(con_weight)
        
class OnlyOneEmb(nn.Module):
    def __init__(self, emb_size:int, embedding_dimension:int=15, context_dimension:int|None=None, init_range:float|None=None, sparse:bool=True, device="cpu"):
        super().__init__()
        self.emb_size:int = emb_size
        self.emb_dim:int = embedding_dimension
        self.word_emb:nn.Embedding = nn.Embedding(num_embeddings=self.emb_size, embedding_dim=self.emb_dim, device=device, sparse=sparse)

        if init_range is None:
            init_range = 0.5 / self.emb_dim
        self.word_emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, centrals_words:list|torch.Tensor, pos_context:list|torch.Tensor, neg_context:list|torch.Tensor):
        words_emb:torch.Tensor = self.word_emb(centrals_words)
        context_emb:torch.Tensor = self.word_emb(pos_context)
        neg_emb:torch.Tensor = self.word_emb(neg_context)

        pos_score = torch.sum(words_emb * context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, words_emb.unsqueeze(-1)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss
    
    def save_weight(self, path:str="SGNS_weights/"):
        word_weights = self.word_emb.weight.detach().cpu()
        torch.save(word_weights, path+'word_embedding.pt')

    def load_weight(self, path:str="SGNS_weights/", name_word_weights:str="word_embedding.pt"):
        word_weights = torch.load(path + name_word_weights)
        self.word_emb:nn.Embedding = nn.Embedding.from_pretrained(word_weights)

# La fonction d’entraînement classique
def train_Word2Vec(modelW2V:nn.Module, dataLoader:Dataset, optimizer:optim.Optimizer, epochs:int, verbal:bool=True, log_interval=100, device="cpu"):
    """Fonction d’entraînement pour un modèle Word2Vec
    """
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        loss_history = []
        global_step = 0
        
        modelW2V.train()

        for batch in dataLoader:
            # centers: [B], pos: [B], negs: [B, K]
            centers, pos, negs = batch
            centers = centers.to(device)
            pos = pos.to(device)
            negs = negs.to(device)

            optimizer.zero_grad()
            loss = modelW2V(centers, pos, negs)
            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            loss_history.append(batch_loss)
            batches += 1
            global_step += 1

            if verbal and log_interval and (global_step % log_interval == 0):
                print(f"Epoch {epoch} Step {global_step} AvgLoss {epoch_loss / batches:.6f}")

        avg_epoch_loss = epoch_loss / max(1, batches)
        if verbal : print(f"Epoch {epoch} finished. Avg loss: {avg_epoch_loss:.6f}")

    return {"loss_history": loss_history, "final_epoch_loss": avg_epoch_loss}
