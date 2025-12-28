
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, causal_mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = x + attn_out
        x = self.ln1(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ln2(x)

        return x


class GPTPathDecoder(nn.Module):
    def __init__(
        self,
        prompt_dim=6,
        hidden_dim=1024,
        n_layers=8,
        n_heads=8,
        prefix_len=4,
        max_T=35,
        pad_value=500
    ):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_T = max_T

        ## add separator token
        self.sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Path token embedding
        self.path_in = nn.Linear(8, hidden_dim)

        self.environment_embed = nn.Linear(4, hidden_dim)  # Example input size


        self.environment_prop_embed = nn.Linear(6, hidden_dim)  # Example input size

        # Positional embeddings
        self.pos_emb = nn.Embedding(max_T + prefix_len, hidden_dim)

        # Convert prompt → prefix tokens
        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)

        # GPT layers
        self.layers = nn.ModuleList([
            GPTBlock(dim=hidden_dim, n_heads=n_heads, ff_dim=4 * hidden_dim)
            for _ in range(n_layers)
        ])

        # Output heads
        self.out = nn.Linear(hidden_dim, 4)  # delay, power, sin(phase), cos(phase)

        # NEW: Multi-label interaction head (4 outputs: R, D, S, T)
        self.interaction_head = nn.Linear(hidden_dim, 4)

        # Path count head
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prompts, paths, interactions, environment, environment_properties, pre_train=False):
        """
        prompts: (B, prompt_dim)
        paths: (B, T, 4)
        interactions: (B,T,4)
        environment: (B, 4)
        environment_properties: (B, T2, 6)
        Returns:
            delay_pred, power_pred, phase_sin_pred, phase_cos_pred,
            phase_pred, pathcounts, interaction_logits
        """

        B, T2, _ = environment_properties.shape
        env_embedding = self.environment_embed(environment).unsqueeze(1)  # (B, 1, hidden_dim)
        env_prop_embedding = self.environment_prop_embed(environment_properties)  # (B, T2, hidden_dim)
        self.env_len = 1 + T2


        B, T, _ = paths.shape

        phase = paths[:, :, 2]
        sinp = torch.sin(phase)
        cosp = torch.cos(phase)

        # Convert prompt → prefix tokens
        if pre_train:
            prefix_raw = self.prompt_to_prefix(prompts * 0.0) # Zero out input
            prefix = prefix_raw.view(B, self.prefix_len, self.hidden_dim)
        else:
            prefix_raw = self.prompt_to_prefix(prompts)
            prefix = prefix_raw.view(B, self.prefix_len, self.hidden_dim)

        # Embed path tokens
        paths_expanded = torch.stack([paths[:, :, 0], paths[:, :, 1], sinp, cosp], dim=-1)



        interactions_clean = interactions.clone()
        interactions_clean[interactions_clean == -1] = 0

        combined = torch.cat([paths_expanded, interactions_clean], dim=-1)
        x = self.path_in(combined)
        # Concatenate prefix + tokens
        dup_sep_token = self.sep_token.expand(B, -1, -1)
        full_seq = torch.cat([env_embedding, dup_sep_token, env_prop_embedding, dup_sep_token, prefix, dup_sep_token, x], dim=1)
        total_len =  self.env_len + self.prefix_len + T + 3  # env + sep + env_prop + sep + prefix + sep + paths
        # Positional embeddings
        pos = self.pos_emb(torch.arange(total_len, device=x.device))
        full_seq = full_seq + pos

        # Causal mask

        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=x.device), 1
        ).bool()

        # Pass through GPT layers
        h = full_seq
        for layer in self.layers:
            h = layer(h, causal_mask)

        # Path predictions
        h_paths = h[:, 3+self.env_len+self.prefix_len:, :]

        # Path parameters
        out = self.out(h_paths)
        delay_pred = out[:, :, 0]
        power_pred = out[:, :, 1]
        phase_sin_pred = out[:, :, 2]
        phase_cos_pred = out[:, :, 3]
        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)

        # NEW: Interaction predictions (multi-label logits)
        interaction_logits = self.interaction_head(h_paths)  # (B, T, 4)

        # Path count head
        prefix_flat = prefix.reshape(B, -1)
        pathcounts = self.pathcount_head(prefix_flat)

        return (delay_pred, power_pred, phase_sin_pred, phase_cos_pred,
                phase_pred, pathcounts, interaction_logits)
    



## encoder model

class PathDecoder(nn.Module):
    def __init__(self, prompt_dim=6, hidden_dim=128, n_layers=4, n_heads=4,  L_max = 25, pad_value=500):
        super().__init__()
        self.pad_value = pad_value

        # Project prompt → conditioning token
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)

        # Path token embedding: delay, power, sin(phase), cos(phase), is_last
        self.path_in = nn.Linear(3, hidden_dim)

        # Positional embedding for sequence steps
        self.pos_emb = nn.Embedding(26, hidden_dim)  # supports up to 25 paths



        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output head: predict next (delay, power, phase_sin, phase_cos, is_stop)
        self.out = nn.Linear(hidden_dim, 4)
        

        self.pathcount_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)   # predict path count
        )

    def forward(self, prompts, paths=None):
        """
        prompts: (B, prompt_dim)
        paths:   (B, T, 4) where columns = [delay_us, power_lin, phase_rad, is_last]
        """
        B, T, _ = paths.shape

        # Convert phase to sin/cos
        phase = paths[:,:,2]
        # sinp = torch.sin(phase)
        # cosp = torch.cos(phase)
 
        # x = torch.stack([paths[:,:,0], paths[:,:,1], sinp, cosp, is_last], dim=-1)
        x = torch.stack([paths[:,:,0], paths[:,:,1], phase], dim=-1)


        # Embed tokens
        x = self.path_in(x)   # (B, T, hidden)
         ## Append SOS embedding

        pos = self.pos_emb(torch.arange(T, device=x.device))  # (T, hidden)

        x = x + pos
        # Conditioning prompt token
        prompt_emb = self.prompt_proj(prompts).unsqueeze(1)   # (B,1,hidden)
       
        

        # Construct causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Decode
        h = self.decoder(
            tgt=x,
            memory=prompt_emb,
            tgt_mask=causal_mask
        )

        # Predict next-step parameters
        out = self.out(h)  # (B, T, 5)

        # return components clearly
        delay_pred = out[:,:,0]
        power_pred = out[:,:,1]
        # phase_sin_pred = out[:,:,2]
        # phase_cos_pred = out[:,:,3]
        # stop_pred = out[:,:,4]
        phase_pred = out[:,:,2]

        stop_pred = out[:,:,3]  # logit for binary cross entropy
        # phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)

        ## path count prediction
        pathcounts = self.pathcount_head( prompt_emb)  #

        return delay_pred, power_pred, phase_pred, pathcounts