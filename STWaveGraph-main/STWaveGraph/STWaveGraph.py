import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import WaveGraphSTNet
from .preprocess import (
    fix_seed,
    preprocess_adata,
    get_expr_feature,
    get_pca_feature,
    construct_st_graphs,
    preprocess_adj_dense,
    spectral_decompose_laplacian_hybrid,
    build_two_feature_views
)
from .utils import (
    barlow_twins_loss,
    cluster_kl_loss,
    cluster_balance_loss,
    graph_smoothness_loss_edge,
    soft_boundary_loss_edge,
    build_edge_index_from_adj,
    cluster_embedding,
    refine_labels_by_spatial_neighbors,
    cluster_consistency_loss 
)


class STWaveGraph:
    """
    ST-domain-friendly Wave-Graph model.

    Key design:
    1. Spatial graph is the backbone
    2. Expression / morphology only reweight spatial edges
    3. Two-stage training:
        - pretrain representation
        - initialize prototypes
        - finetune clustering
    4. Final prediction uses embedding clustering + spatial refinement
    """
    def __init__(
        self,
        adata,
        n_clusters,
        device=None,

        # preprocessing
        n_top_genes=2000,
        pca_dim=50,
        counts_layer=None,
        random_seed=42,

        # graph
        n_neighbors_spatial=6,
        use_morphology=False,
        morph_key="image_feat",
        expr_power=1.0,
        morph_power=1.0,

        # model
        hidden_dim=64,
        proj_dim=32,
        n_bands=3,
        dropout=0.1,
        spectral_k=128,
        spectral_low_ratio=0.75,

        # training
        pretrain_epochs=300,
        finetune_epochs=150,
        lr=1e-3,
        finetune_lr=5e-4,
        weight_decay=1e-4,
        feature_mask_rate=0.15,
        feature_noise_std=0.1,
        regularization_warmup_epochs=20,

        # losses
        lambda_rec=0.2,
        lambda_ssl=1.0,
        lambda_smooth=0.05,
        lambda_kl=1.0,
        lambda_balance=0.05,
        lambda_boundary=0.1,
        boundary_margin=1.0,
        lambda_q_cons=0.5, 
        use_q_for_pred=False, # 默认值为 False，保持和你之前的行为一致

        # inference / clustering
        cluster_method="mclust",
        cluster_pca_dim=20,
        mclust_model="EEE",
        mclust_fallback_to_gmm=True,
        refine_iters=2
    ):
        self.adata = adata.copy()
        self.n_clusters = n_clusters
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_top_genes = n_top_genes
        self.pca_dim = pca_dim
        self.counts_layer = counts_layer
        self.random_seed = random_seed

        self.n_neighbors_spatial = n_neighbors_spatial
        self.use_morphology = use_morphology
        self.morph_key = morph_key
        self.expr_power = expr_power
        self.morph_power = morph_power

        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.n_bands = n_bands
        self.dropout = dropout
        self.spectral_k = spectral_k
        self.spectral_low_ratio = spectral_low_ratio

        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        self.feature_mask_rate = feature_mask_rate
        self.feature_noise_std = feature_noise_std
        self.regularization_warmup_epochs = regularization_warmup_epochs

        self.lambda_rec = lambda_rec
        self.lambda_ssl = lambda_ssl
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl
        self.lambda_balance = lambda_balance
        self.lambda_boundary = lambda_boundary
        self.boundary_margin = boundary_margin
        self.lambda_q_cons = lambda_q_cons 
        self.use_q_for_pred = use_q_for_pred

        self.cluster_method = cluster_method
        self.cluster_pca_dim = cluster_pca_dim
        self.mclust_model = mclust_model
        self.mclust_fallback_to_gmm = mclust_fallback_to_gmm
        self.refine_iters = refine_iters

        fix_seed(self.random_seed)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        # 1. preprocess
        self.adata = preprocess_adata(
            self.adata,
            n_top_genes=self.n_top_genes,
            counts_layer=self.counts_layer
        )

        # 2. gene features
        self.expr_feat = get_expr_feature(self.adata, use_hvg=True)
        self.x_pca, self.pca_model = get_pca_feature(
            self.adata,
            n_pcs=self.pca_dim,
            use_hvg=True,
            random_state=self.random_seed
        )

        # 3. graphs
        graph_dict = construct_st_graphs(
            adata=self.adata,
            expr_feat_for_graph=self.x_pca,
            n_neighbors_spatial=self.n_neighbors_spatial,
            use_morphology=self.use_morphology,
            morph_key=self.morph_key,
            expr_power=self.expr_power,
            morph_power=self.morph_power
        )

        self.adj_spatial_raw = graph_dict["adj_spatial"]
        self.adj_wave_raw = graph_dict["adj_wave"]

        # local branch uses spatial graph
        self.adj_local = preprocess_adj_dense(self.adj_spatial_raw, add_self_loop=True)

        # wave branch uses spectral decomposition on wave graph
        self.eigvals, self.eigvecs = spectral_decompose_laplacian_hybrid(
            self.adj_wave_raw,
            k=self.spectral_k,
            low_ratio=self.spectral_low_ratio
        )

        # edge list for spatial regularization / refinement
        edge_index, edge_weight = build_edge_index_from_adj(
            self.adj_spatial_raw,
            threshold=0.0,
            add_self_loop=False
        )

        # tensors
        self.x = torch.FloatTensor(self.x_pca).to(self.device)
        self.adj_local_t = torch.FloatTensor(self.adj_local).to(self.device)
        self.eigvals_t = torch.FloatTensor(self.eigvals).to(self.device)
        self.eigvecs_t = torch.FloatTensor(self.eigvecs).to(self.device)
        self.edge_index_t = edge_index.to(self.device)
        self.edge_weight_t = edge_weight.to(self.device)

    def _build_model(self):
        self.model = WaveGraphSTNet(
            in_dim=self.x.shape[1],
            hidden_dim=self.hidden_dim,
            n_clusters=self.n_clusters,
            proj_dim=self.proj_dim,
            dropout=self.dropout,
            n_bands=self.n_bands
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _reset_optimizer_for_finetune(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.finetune_lr,
            weight_decay=self.weight_decay
        )

    def _build_two_views(self):
        x1_np, x2_np = build_two_feature_views(
            self.x_pca,
            mask_rate=self.feature_mask_rate,
            noise_std=self.feature_noise_std
        )
        x1 = torch.FloatTensor(x1_np).to(self.device)
        x2 = torch.FloatTensor(x2_np).to(self.device)
        return x1, x2

    def _pretrain_step(self):
        self.model.train()

        x1, x2 = self._build_two_views()

        h1, h2, z1, z2, rec1, rec2, q1, q2 = self.model(
            x1, x2, self.adj_local_t, self.eigvecs_t, self.eigvals_t
        )

        loss_rec = 0.5 * (
            F.mse_loss(rec1, self.x) +
            F.mse_loss(rec2, self.x)
        )

        loss_ssl = barlow_twins_loss(z1, z2)


        h1_n = F.normalize(h1, p=2, dim=1)
        h2_n = F.normalize(h2, p=2, dim=1)

        loss_smooth = 0.5 * (
            graph_smoothness_loss_edge(h1_n, self.edge_index_t, self.edge_weight_t) +
            graph_smoothness_loss_edge(h2_n, self.edge_index_t, self.edge_weight_t)
        )

        loss = (
            self.lambda_rec * loss_rec +
            self.lambda_ssl * loss_ssl +
            self.lambda_smooth * loss_smooth
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_rec": loss_rec.item(),
            "loss_ssl": loss_ssl.item(),
            "loss_smooth": loss_smooth.item()
        }

    @torch.no_grad()
    def _initialize_cluster_centers(self):
        """
        Prototype initialization in the same normalized embedding space
        used by the clustering head.
        """
        self.model.eval()
        h, z, rec, q = self.model.inference(
            self.x, self.adj_local_t, self.eigvecs_t, self.eigvals_t
        )

        emb = F.normalize(h, p=2, dim=1).cpu().numpy()
        init_labels, init_probs, init_centers = cluster_embedding(
            emb,
            n_clusters=self.n_clusters,
            method="kmeans",
            random_state=self.random_seed,
            pca_dim=None
        )

        centers_t = torch.FloatTensor(init_centers).to(self.device)
        self.model.cluster_head.initialize_centers(centers_t)

    def _finetune_step(self, epoch):
        self.model.train()

        x1, x2 = self._build_two_views()

        h1, h2, z1, z2, rec1, rec2, q1, q2 = self.model(
            x1, x2, self.adj_local_t, self.eigvecs_t, self.eigvals_t
        )

        loss_rec = 0.5 * (
            F.mse_loss(rec1, self.x) +
            F.mse_loss(rec2, self.x)
        )

        loss_ssl = barlow_twins_loss(z1, z2)


        h1_n = F.normalize(h1, p=2, dim=1)
        h2_n = F.normalize(h2, p=2, dim=1)

        loss_smooth = 0.5 * (
            graph_smoothness_loss_edge(h1_n, self.edge_index_t, self.edge_weight_t) +
            graph_smoothness_loss_edge(h2_n, self.edge_index_t, self.edge_weight_t)
        )
        loss_q_cons = cluster_consistency_loss(q1, q2)




        loss_kl = 0.5 * (
            cluster_kl_loss(q1) +
            cluster_kl_loss(q2)
        )

        loss_balance = 0.5 * (
            cluster_balance_loss(q1) +
            cluster_balance_loss(q2)
        )
        loss_boundary = 0.5 * (
            soft_boundary_loss_edge(
                h1_n, q1, self.edge_index_t, self.edge_weight_t,
                margin=self.boundary_margin,
                detach_q=False
            ) +
            soft_boundary_loss_edge(
                h2_n, q2, self.edge_index_t, self.edge_weight_t,
                margin=self.boundary_margin,
                detach_q=False
            )
        )

        # warmup: use only core objectives first
        use_extra_reg = (epoch + 1) > self.regularization_warmup_epochs
        extra_scale = 1.0 if use_extra_reg else 0.0
        loss = (
            self.lambda_rec * loss_rec +
            self.lambda_ssl * loss_ssl +
            self.lambda_smooth * loss_smooth +
            self.lambda_kl * loss_kl +
            self.lambda_q_cons * loss_q_cons +
            extra_scale * (
                self.lambda_balance * loss_balance +
                self.lambda_boundary * loss_boundary
            )
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_rec": loss_rec.item(),
            "loss_ssl": loss_ssl.item(),
            "loss_smooth": loss_smooth.item(),
            "loss_kl": loss_kl.item(),
            "loss_balance": loss_balance.item(),
            "loss_boundary": loss_boundary.item(),
            "extra_reg_on": float(use_extra_reg)
        }

    def fit(self, verbose=True):
        self.history_pretrain = []
        self.history_finetune = []

        if verbose:
            print("========== Stage 1: Representation Pretraining ==========")

        for epoch in tqdm(range(self.pretrain_epochs), disable=not verbose):
            log = self._pretrain_step()
            log["epoch"] = epoch + 1
            self.history_pretrain.append(log)

        if verbose:
            print("========== Initialize Cluster Centers ==========")
        self._initialize_cluster_centers()
        self._reset_optimizer_for_finetune()

        if verbose:
            print("========== Stage 2: Clustering Finetuning ==========")

        for epoch in tqdm(range(self.finetune_epochs), disable=not verbose):
            log = self._finetune_step(epoch)
            log["epoch"] = epoch + 1
            self.history_finetune.append(log)

            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"[Finetune] Epoch {epoch+1}/{self.finetune_epochs} | "
                    f"loss={log['loss']:.4f} | "
                    f"rec={log['loss_rec']:.4f} | "
                    f"ssl={log['loss_ssl']:.4f} | "
                    f"smooth={log['loss_smooth']:.4f} | "
                    f"kl={log['loss_kl']:.4f} | "
                    f"balance={log['loss_balance']:.4f} | "
                    f"bd={log['loss_boundary']:.4f} | "
                    f"extra_reg_on={int(log['extra_reg_on'])}"
                )

        return {
            "pretrain": self.history_pretrain,
            "finetune": self.history_finetune
        }

    @torch.no_grad()
    def infer(self, refine=True):
        self.model.eval()

        h, z, rec, q = self.model.inference(
            self.x, self.adj_local_t, self.eigvecs_t, self.eigvals_t
        )

        emb = F.normalize(h, p=2, dim=1).cpu().numpy().astype(np.float32)
        q_np = q.cpu().numpy().astype(np.float32)

        if self.use_q_for_pred:
            print("INFO: Using model's internal clustering head (q) for prediction.")
            labels_raw = np.argmax(q_np, axis=1).astype(np.int32)
            probs = q_np
            cluster_method_used = "model_prototype_head"
            cluster_pca_dim_used = "N/A"
            mclust_model_used = "N/A"
        else:
            print(f"INFO: Using external clustering ('{self.cluster_method}') on embedding (h) for prediction.")
            labels_raw, probs, _ = cluster_embedding(
                emb,
                n_clusters=self.n_clusters,
                method=self.cluster_method,
                random_state=self.random_seed,
                pca_dim=self.cluster_pca_dim,
                mclust_model=self.mclust_model,
                fallback_to_gmm=self.mclust_fallback_to_gmm
            )
            cluster_method_used = self.cluster_method 
            cluster_pca_dim_used = self.cluster_pca_dim
            mclust_model_used = self.mclust_model

        if refine:
            labels = refine_labels_by_spatial_neighbors(
                labels_raw,
                self.adj_spatial_raw,
                n_iter=self.refine_iters,
                self_weight=1.0
            )
        else:
            labels = labels_raw.copy()

        self.adata.obsm["emb"] = emb
        self.adata.obsm["model_q"] = q_np
        self.adata.obsm["soft_assign"] = probs.astype(np.float32)
        self.adata.obs["pred_domain_raw"] = labels_raw.astype(str)
        self.adata.obs["pred_domain"] = labels.astype(str)
        self.adata.uns["use_q_for_pred"] = self.use_q_for_pred 
        self.adata.uns["cluster_method_used"] = cluster_method_used
        self.adata.uns["cluster_pca_dim"] = cluster_pca_dim_used
        self.adata.uns["mclust_model"] = mclust_model_used
        self.adata.uns["regularization_warmup_epochs"] = self.regularization_warmup_epochs

        return self.adata

    def train(self, verbose=True):
        self.fit(verbose=verbose)
        return self.infer(refine=True)