"""
RC-DGAT Model Architecture
Regime-Conditioned Dynamic Graph Attention Network for Stock Return Forecasting by Devender
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hmmlearn import hmm
from scipy.stats import zscore
from config import (
    HMM_CONFIG, GRAPH_CONFIG, REGIME_CONFIG, GAT_CONFIG, 
    GRU_CONFIG, PREDICTION_HEAD_CONFIG
)


class HMMRegimeDetector(nn.Module):
    """
    Hidden Markov Model for market regime detection.
    Detects Bull, Bear, and High-Volatility states from return-volatility observations.
    """
    
    def __init__(self, config=HMM_CONFIG):
        super(HMMRegimeDetector, self).__init__()
        self.config = config
        self.hmm_model = None
        self.states = config['state_names']
        self.num_states = config['num_states']
        
    def fit(self, returns, volatilities):
        """
        Fit HMM on return-volatility pairs.
        
        Args:
            returns: (T,) or (T, N) array of daily returns (if 2D, uses mean across stocks)
            volatilities: (T,) or (T, N) array of rolling volatilities (if 2D, uses mean across stocks)
        """
        # Handle 2D arrays by taking mean across stocks
        if returns.ndim == 2:
            returns = returns.mean(axis=1)
        if volatilities.ndim == 2:
            volatilities = volatilities.mean(axis=1)
        
        # Prepare observation matrix: combine returns and volatilities
        observations = np.column_stack([returns, volatilities])
        observations = zscore(observations, axis=0)  # Normalize
        # Ensure observations is a numpy array for len()
        observations = np.asarray(observations)
        # Ensure minimum samples (require at least 100 observations)
        if len(observations) < 100:
            # For testing with small data, use fewer iterations and add regularization
            n_iter = max(10, self.config['n_iter'] // 10)
        else:
            n_iter = self.config['n_iter']
        
        # Fit Gaussian HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.num_states,
            covariance_type=self.config['covariance_type'],
            n_iter=n_iter,
            random_state=self.config['random_state'],
            tol=1e-4  # Add convergence tolerance
        )
        
        try:
            self.hmm_model.fit(observations)
            # Ensure transition matrix is valid
            if np.any(self.hmm_model.transmat_.sum(axis=1) != 1.0):
                # Re-normalize if needed
                self.hmm_model.transmat_ = self.hmm_model.transmat_ / self.hmm_model.transmat_.sum(axis=1, keepdims=True)
        except Exception as e:
            print(f"Warning: HMM fitting encountered an issue: {e}")
        
        return self
    
    def predict_states(self, returns, volatilities):
        """
        Predict regime states.
        by devender
        Args:
            returns: (T,) or (T, N) array of daily returns (if 2D, uses mean across stocks)
            volatilities: (T,) or (T, N) array of rolling volatilities (if 2D, uses mean across stocks)
            
        Returns:
            states: (T,) array of regime state indices
        """
        # Handle 2D arrays by taking mean across stocks (same as fit)
        if returns.ndim == 2:
            returns = returns.mean(axis=1)
        if volatilities.ndim == 2:
            volatilities = volatilities.mean(axis=1)
        
        # Check for NaN values and replace with 0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.1, neginf=-0.1)
        volatilities = np.nan_to_num(volatilities, nan=0.01, posinf=0.1, neginf=0.01)
        
        observations = np.column_stack([returns, volatilities])
        
        # Standardize but handle edge cases
        obs_mean = observations.mean(axis=0)
        obs_std = observations.std(axis=0) + 1e-8  # Prevent division by zero
        observations = (observations - obs_mean) / obs_std
        
        if self.hmm_model is None:
            raise RuntimeError("HMM model is not fitted. Call 'fit' before predicting states.")
        states = self.hmm_model.predict(observations)
        return states
    
    def predict_proba(self, returns, volatilities):
        """
        Get probability distribution over regime states.
        
        Args:
            returns: (T,) or (T, N) array of daily returns (if 2D, uses mean across stocks)
            volatilities: (T,) or (T, N) array of rolling volatilities (if 2D, uses mean across stocks)
        
        Returns:
            state_probs: (T, num_states) array of state probabilities
        """
        # Handle 2D arrays by taking mean across stocks (same as fit)
        if returns.ndim == 2:
            returns = returns.mean(axis=1)
        if volatilities.ndim == 2:
            volatilities = volatilities.mean(axis=1)
        
        # Check for NaN values and replace with safe defaults
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.1, neginf=-0.1)
        volatilities = np.nan_to_num(volatilities, nan=0.01, posinf=0.1, neginf=0.01)
        
        observations = np.column_stack([returns, volatilities])
        
        # Standardize but handle edge cases
        obs_mean = observations.mean(axis=0)
        obs_std = observations.std(axis=0) + 1e-8  # Prevent division by zero
        observations = (observations - obs_mean) / obs_std
        
        # Get state probabilities using posteriors
        try:
            if self.hmm_model is None:
                raise RuntimeError("HMM model is not fitted. Call 'fit' before predicting probabilities.")
            # Use predict method to get states
            states = self.hmm_model.predict(observations)
            # Create probability matrix: 1.0 for predicted state, 0 for others
            state_probs = np.zeros((len(states), self.num_states))
            state_probs[np.arange(len(states)), states] = 1.0
        except Exception:
            # Fallback: uniform probabilities
            state_probs = np.ones((len(observations), self.num_states)) / self.num_states
        
        return state_probs
    
    def forward(self, returns, volatilities):
        """Forward pass: return regime embeddings"""
        if self.hmm_model is None:
            raise ValueError("HMM model not fitted. Call fit() first.")
        
        # Convert to numpy if tensors
        if torch.is_tensor(returns):
            returns_np = returns.cpu().numpy()
        else:
            returns_np = returns
            
        if torch.is_tensor(volatilities):
            volatilities_np = volatilities.cpu().numpy()
        else:
            volatilities_np = volatilities
        
        states = self.predict_states(returns_np, volatilities_np)
        state_probs = self.predict_proba(returns_np, volatilities_np)
        
        return torch.from_numpy(states).long(), torch.from_numpy(state_probs).float()


class DynamicGraphConstructor(nn.Module):
    """
    Constructs dynamic graph based on rolling correlations.
    Graph topology and edge weights are adjusted based on detected market regime.
    """
    
    def __init__(self, config=GRAPH_CONFIG):
        super(DynamicGraphConstructor, self).__init__()
        self.config = config
        self.window_size = config['window_size']
        self.threshold = config['threshold']
        self.correlation_type = config['correlation_type']
        
    def compute_correlation_matrix(self, price_data):
        """
        Compute correlation matrix from price data.
        
        Args:
            price_data: (batch, num_stocks, features)
            
        Returns:
            corr_matrix: (num_stocks, num_stocks)
        """
        batch_size, num_stocks, _ = price_data.shape
        
        # Use closing prices (last feature) and aggregate across batch
        closes = price_data[:, :, -1]  # (batch, num_stocks)
        closes_flat = closes.reshape(-1, num_stocks)  # (batch, num_stocks)
        
        # Compute correlation matrix across all time periods
        # Transpose to get (num_stocks, time_periods)
        closes_t = closes_flat.t()  # (num_stocks, batch)
        
        # Compute Pearson correlation
        mean = closes_t.mean(dim=1, keepdim=True)  # (num_stocks, 1)
        std = closes_t.std(dim=1, keepdim=True)  # (num_stocks, 1)
        
        # Normalize
        normalized = (closes_t - mean) / (std + 1e-8)  # (num_stocks, batch)
        
        # Correlation matrix
        corr_matrix = torch.mm(normalized, normalized.t()) / batch_size  # (num_stocks, num_stocks)
        
        return corr_matrix
    
    def apply_regime_modulation(self, corr_matrix, regime_states):
        """
        Modulate correlation matrix based on market regime.
        
        Args:
            corr_matrix: (num_stocks, num_stocks)
            regime_states: (batch,) regime state indices
            
        Returns:
            modulated_corr: (num_stocks, num_stocks)
        """
        regime_weights = self.config['regime_weight_dict']
        
        # Get dominant regime
        state_names = self.config.get('state_names', ['Bull', 'Bear', 'HighVolatility'])
        dominant_regime = state_names[regime_states.mode().values.item()]
        
        # Apply regime-specific scaling
        scale = regime_weights[dominant_regime]['correlation_scale']
        modulated_corr = corr_matrix * scale
        modulated_corr = torch.clamp(modulated_corr, -1.0, 1.0)
        
        return modulated_corr
    
    def create_edge_index(self, corr_matrix, threshold=None):
        """
        Create edge indices from correlation matrix.
        
        Args:
            corr_matrix: (num_stocks, num_stocks)
            threshold: Correlation threshold for edges
            
        Returns:
            edge_index: (2, num_edges) tensor of edge indices
            edge_attr: (num_edges,) tensor of edge weights
        """
        if threshold is None:
            threshold = self.threshold
        
        num_stocks = corr_matrix.shape[0]
        
        # Create edges for correlations above threshold
        edges = []
        edge_weights = []
        
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                if abs(corr_matrix[i, j]) > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    edge_weights.append(corr_matrix[i, j])
                    edge_weights.append(corr_matrix[i, j])
        
        if len(edges) == 0:
            # Ensure at least some edges exist (top-K neighbors)
            k = max(3, int(0.1 * num_stocks))
            for i in range(num_stocks):
                top_k = torch.topk(torch.abs(corr_matrix[i]), k + 1)[1][1:]
                for j in top_k:
                    edges.append([i, j.item()])
                    edge_weights.append(corr_matrix[i, j].item())
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # Fallback: complete graph
            edge_index = torch.ones((2, 0), dtype=torch.long)
            edge_attr = torch.ones((0,), dtype=torch.float)
        
        return edge_index, edge_attr
    
    def forward(self, price_data, regime_states):
        """
        Construct dynamic graph.
        
        Returns:
            edge_index: (2, num_edges)
            edge_attr: (num_edges,)
        """
        corr_matrix = self.compute_correlation_matrix(price_data)
        modulated_corr = self.apply_regime_modulation(corr_matrix, regime_states)
        edge_index, edge_attr = self.create_edge_index(modulated_corr)
        
        return edge_index, edge_attr


class RegimeEmbedding(nn.Module):
    """
    Learned embedding for market regimes.
    """
    
    def __init__(self, num_states=3, embedding_dim=16):
        super(RegimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)
        
    def forward(self, regime_indices):
        """
        Args:
            regime_indices: (batch,) or (T,) indices of regimes
            
        Returns:
            embeddings: (batch, embedding_dim) or (T, embedding_dim)
        """
        return self.embedding(regime_indices)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer.
    Computes attention-weighted aggregation of node neighborhoods.
    """
    
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.3):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Linear transformations for queries, keys, values
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        
        # Attention coefficients
        self.attn_weight = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim, 1))
        nn.init.xavier_uniform_(self.attn_weight)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, in_dim) node features
            edge_index: (2, num_edges) edge indices
            edge_attr: (num_edges,) optional edge weights
            
        Returns:
            out: (num_nodes, out_dim) updated node features
        """
        num_nodes = x.shape[0]
        
        # Linear transformations
        Q = self.query(x)  # (num_nodes, out_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores for edges
        out_list = []
        
        for h in range(self.num_heads):
            # Initialize output for this head
            head_out = torch.zeros((num_nodes, self.head_dim), 
                                   device=x.device, dtype=x.dtype)
            
            # Process each destination node
            for i in range(num_nodes):
                mask = edge_index[1] == i
                if mask.sum() > 0:
                    src_indices = edge_index[0][mask]
                    
                    # Get scores for edges ending at node i
                    q_i = Q[i, h]  # (head_dim,)
                    k_src = K[src_indices, h]  # (num_edges_to_i, head_dim)
                    
                    # Compute attention scores
                    scores = torch.matmul(k_src, q_i)  # (num_edges_to_i,)
                    scores = self.leaky_relu(scores)
                    
                    # Softmax
                    attn_weights = F.softmax(scores, dim=0)  # (num_edges_to_i,)
                    
                    # Aggregate values
                    v_src = V[src_indices, h]  # (num_edges_to_i, head_dim)
                    head_out[i] = torch.sum(attn_weights.unsqueeze(1) * v_src, dim=0)
            
            out_list.append(head_out)
        
        # Concatenate heads
        out = torch.cat(out_list, dim=1)  # (num_nodes, out_dim)
        out = self.dropout(out)
        
        return out


class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention Network.
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, dropout=0.3, 
                 num_layers=2):
        super(MultiHeadGAT, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout))
        
        # Output layer
        self.layers.append(GraphAttentionLayer(hidden_dim, out_dim, num_heads, dropout))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges,)
            
        Returns:
            out: (num_nodes, out_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        return x


class RegimeAwareGRU(nn.Module):
    """
    GRU with regime-aware gating.
    Regime information modulates the forget and reset gates.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, 
                 regime_embedding_dim=16):
        super(RegimeAwareGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.regime_embedding_dim = regime_embedding_dim
        
        # Standard GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Regime gating
        self.regime_gate_layer = nn.Sequential(
            nn.Linear(regime_embedding_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, h=None, regime_embeddings=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            h: (num_layers, batch, hidden_dim) initial hidden state
            regime_embeddings: (seq_len, batch, regime_embedding_dim)
            
        Returns:
            out: (batch, seq_len, hidden_dim)
            h: (num_layers, batch, hidden_dim)
        """
        # Standard GRU forward pass
        out, h = self.gru(x, h)
        
        # Apply regime-aware gating
        if regime_embeddings is not None:
            batch_size = out.shape[0]
            seq_len = out.shape[1]
            
            # Create new output tensor to avoid in-place operations
            out_gated = []
            
            for t in range(seq_len):
                regime_emb = regime_embeddings[:, t, :]  # (batch, regime_embedding_dim)
                h_t = out[:, t, :]  # (batch, hidden_dim)
                
                # Combine regime embedding with hidden state
                combined = torch.cat([regime_emb, h_t], dim=1)
                gate = self.regime_gate_layer(combined)
                
                # Apply gating
                gated_t = gate * h_t
                out_gated.append(gated_t)
            
            out = torch.stack(out_gated, dim=1)  # (batch, seq_len, hidden_dim)
        
        return out, h


class MLPHead(nn.Module):
    """
    Multi-layer perceptron prediction head.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rates):
        super(MLPHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
            
        Returns:
            out: (batch, output_dim)
        """
        return self.net(x)


class RCDGAT(nn.Module):
    """
    RC-DGAT: Regime-Conditioned Dynamic Graph Attention Network.
    Complete model combining all components.
    """
    
    def __init__(self):
        super(RCDGAT, self).__init__()
        
        # Regime detection
        self.hmm_detector = HMMRegimeDetector(HMM_CONFIG)
        self.regime_embedding = RegimeEmbedding(
            num_states=HMM_CONFIG['num_states'],
            embedding_dim=REGIME_CONFIG['embedding_dim']
        )
        
        # Graph construction
        self.graph_constructor = DynamicGraphConstructor(GRAPH_CONFIG)
        
        # Graph attention
        self.gat = MultiHeadGAT(
            in_dim=GAT_CONFIG['input_dim'],
            hidden_dim=GAT_CONFIG['hidden_dim'],
            out_dim=GAT_CONFIG['output_dim'],
            num_heads=GAT_CONFIG['num_heads'],
            dropout=GAT_CONFIG['dropout'],
            num_layers=2
        )
        
        # Temporal encoder
        self.gru = RegimeAwareGRU(
            input_dim=GAT_CONFIG['output_dim'],
            hidden_dim=GRU_CONFIG['hidden_dim'],
            num_layers=GRU_CONFIG['num_layers'],
            dropout=GRU_CONFIG['dropout'],
            regime_embedding_dim=REGIME_CONFIG['embedding_dim']
        )
        
        # Prediction head
        self.mlp_head = MLPHead(
            input_dim=GRU_CONFIG['hidden_dim'],
            hidden_dims=PREDICTION_HEAD_CONFIG['hidden_dims'],
            output_dim=PREDICTION_HEAD_CONFIG['output_dim'],
            dropout_rates=PREDICTION_HEAD_CONFIG['dropout']
        )
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(GRU_CONFIG['hidden_dim'], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, HMM_CONFIG['num_states'])
        )
    
    def forward(self, price_data, returns, volatilities):
        """
        Forward pass through the entire model.
        
        Args:
            price_data: (batch, seq_len, num_stocks, features) or (batch, num_stocks, features)
            returns: (batch, seq_len, num_stocks) or (batch, num_stocks) daily returns
            volatilities: (batch, seq_len, num_stocks) or (batch, num_stocks) rolling volatilities
            
        Returns:
            predictions: (batch, num_stocks, 1) return forecasts
            regime_logits: (batch, num_states) regime classification logits
            regime_states: (batch,) detected regime states
        """
        # Handle sequence dimension: use last timestep
        if price_data.ndim == 4:  # (batch, seq_len, num_stocks, features)
            price_data = price_data[:, -1, :, :]  # (batch, num_stocks, features)
        if returns.ndim == 3:  # (batch, seq_len, num_stocks)
            returns = returns[:, -1, :]  # (batch, num_stocks)
        if volatilities.ndim == 3:  # (batch, seq_len, num_stocks)
            volatilities = volatilities[:, -1, :]  # (batch, num_stocks)
        
        batch_size = price_data.shape[0]
        num_stocks = price_data.shape[1]
        
        # Detect market regimes
        regime_states, regime_probs = self.hmm_detector(returns, volatilities)
        
        # Regime embeddings
        regime_emb = self.regime_embedding(regime_states)  # (batch, embedding_dim)
        
        # Construct dynamic graph
        edge_index, edge_attr = self.graph_constructor(price_data, regime_states)
        edge_index = edge_index.to(price_data.device)
        edge_attr = edge_attr.to(price_data.device)
        
        # Graph attention on stock features
        stock_features = price_data[:, :, :GAT_CONFIG['input_dim']]
        
        gat_out_list = []
        for b in range(batch_size):
            gat_out = self.gat(stock_features[b], edge_index, edge_attr)
            gat_out_list.append(gat_out)
        
        gat_out = torch.stack(gat_out_list)  # (batch, num_stocks, output_dim)
        
        # Temporal encoding with regime modulation
        # Expand regime embedding to match sequence length (one regime state per stock)
        regime_emb_expanded = regime_emb.unsqueeze(1).expand(-1, gat_out.shape[1], -1)  # (batch, num_stocks, embedding_dim)
        
        gru_out, _ = self.gru(gat_out, regime_embeddings=regime_emb_expanded)
        
        # Prediction
        predictions = self.mlp_head(gru_out)  # (batch, num_stocks, 1)
        
        # Regime classification
        regime_logits = self.regime_classifier(gru_out[:, 0, :])  # Use first timestep
        
        return predictions, regime_logits, regime_states


if __name__ == "__main__":
    # Example usage
    model = RCDGAT()
    print(model)
    
    # Create dummy input
    batch_size = 4
    num_stocks = 50
    seq_len = 20
    num_features = 31
    
    price_data = torch.randn(batch_size, num_stocks, num_features)
    returns = torch.randn(batch_size, num_stocks)
    volatilities = torch.abs(torch.randn(batch_size, num_stocks))
    
    # Forward pass
    predictions, regime_logits, regime_states = model(price_data, returns, volatilities)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Regime logits shape: {regime_logits.shape}")
    print(f"Regime states shape: {regime_states.shape}")
