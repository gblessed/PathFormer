
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
        hidden_dim=512,
        n_layers=6,
        n_heads=4,
        prefix_len=4,
        max_T=26,
        pad_value=500
    ):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_T = max_T

        # Path token embedding
        # Now: [delay, power, sin(phase), cos(phase), sin(az), cos(az), sin(el), cos(el)] => 8
        # plus interactions (4) => combined input dim = 12
        self.path_in = nn.Linear(12, hidden_dim)

        self.pos_emb = nn.Embedding(max_T + prefix_len, hidden_dim)

        # Convert prompt → prefix tokens
        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)

        # GPT layers
        self.layers = nn.ModuleList([
            GPTBlock(dim=hidden_dim, n_heads=n_heads, ff_dim=4 * hidden_dim)
            for _ in range(n_layers)
        ])

        # Output heads
        # outputs: delay, power, sin(phase), cos(phase), sin(az), cos(az), sin(el), cos(el)
        self.out = nn.Linear(hidden_dim, 8)

        # NEW: Multi-label interaction head (4 outputs: R, D, S, T)
        self.interaction_head = nn.Linear(hidden_dim, 4)

        # Path count head
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prompts, paths, interactions):
        """
        prompts: (B, prompt_dim)
        paths: (B, T, 5)  # delay, power, phase, aoa_az, aoa_el
        interactions: (B, T, 4)
        Returns:
            delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
            az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
            pathcounts, interaction_logits
        """
        B, T, _ = paths.shape

        phase = paths[:, :, 2]
        sinp = torch.sin(phase)
        cosp = torch.cos(phase)

        aoa_az = paths[:, :, 3]
        sin_az = torch.sin(aoa_az)
        cos_az = torch.cos(aoa_az)

        aoa_el = paths[:, :, 4]
        sin_el = torch.sin(aoa_el)
        cos_el = torch.cos(aoa_el)
        
        # Convert prompt → prefix tokens
        prefix_raw = self.prompt_to_prefix(prompts)
        prefix = prefix_raw.view(B, self.prefix_len, self.hidden_dim)

        # Embed path tokens: delay, power, sin(phase), cos(phase), sin(az), cos(az), sin(el), cos(el)
        paths_expanded = torch.stack([
            paths[:, :, 0],  # delay
            paths[:, :, 1],  # power
            sinp, cosp,
            sin_az, cos_az,
            sin_el, cos_el
        ], dim=-1)
        

        interactions_clean = interactions.clone()
        interactions_clean[interactions_clean == -1] = 0
    
        combined = torch.cat([paths_expanded, interactions_clean], dim=-1)
        x = self.path_in(combined)
        # Concatenate prefix + tokens
        full_seq = torch.cat([prefix, x], dim=1)

        # Positional embeddings
        pos = self.pos_emb(torch.arange(self.prefix_len + T, device=x.device))
        full_seq = full_seq + pos

        # Causal mask
        total_len = self.prefix_len + T
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=x.device), 1
        ).bool()

        # Pass through GPT layers
        h = full_seq
        for layer in self.layers:
            h = layer(h, causal_mask)

        # Path predictions
        h_paths = h[:, self.prefix_len:, :]

        # Path parameters
        out = self.out(h_paths)
        delay_pred = out[:, :, 0]
        power_pred = out[:, :, 1]
        phase_sin_pred = out[:, :, 2]
        phase_cos_pred = out[:, :, 3]
        az_sin_pred = out[:, :, 4]
        az_cos_pred = out[:, :, 5]
        el_sin_pred = out[:, :, 6]
        el_cos_pred = out[:, :, 7]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)

        # NEW: Interaction predictions (multi-label logits)
        interaction_logits = self.interaction_head(h_paths)  # (B, T, 4)

        # Path count head
        prefix_flat = prefix.reshape(B, -1)
        pathcounts = self.pathcount_head(prefix_flat)

        return (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
            az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
            pathcounts, interaction_logits)

class PathDecoder(nn.Module):
    def __init__(self, prompt_dim=6, hidden_dim=512, n_layers=6, n_heads=4,  max_T = 35, prefix_len=4, pad_value=500):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_T = max_T
        # Project prompt → conditioning token
        # self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)
        # Path token embedding: delay, power, sin(phase), cos(phase), is_last
        self.path_in = nn.Linear(12, hidden_dim)

        # Positional embedding for sequence steps
        self.pos_emb = nn.Embedding(max_T, hidden_dim)  # supports up to 25 paths

        # self.environment_embed = nn.Linear(4, hidden_dim)  # Example input size
        # self.environment_prop_embed = nn.Linear(6, hidden_dim)
        
        self.interaction_head = nn.Linear(hidden_dim, 4)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output head: predict next (delay, power, phase_sin, phase_cos, is_stop)
        # self.out = nn.Linear(hidden_dim, 8)
        
        self.out = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 8)
                )
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, prompts, paths, interactions):
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

        # B, T2, _ = environment_properties.shape
        # env_embedding = self.environment_embed(environment).unsqueeze(1)  # (B, 1, hidden_dim)
        # env_prop_embedding = self.environment_prop_embed(environment_properties)  # (B, T2, hidden_dim)
        # self.env_len = 1 + T2

        # Convert prompt → prefix tokens
        # if pre_train:
        #     prefix_raw = self.prompt_to_prefix(prompts * 0.0) # Zero out input
        #     prefix = torch.zeros( (B, self.prefix_len, self.hidden_dim)).to("cuda")
        # else:



        B, T, _ = paths.shape
        prefix_raw = self.prompt_to_prefix(prompts)
        prefix = prefix_raw.view(B, self.prefix_len, self.hidden_dim)

        # Convert phase to sin/cos
        phase = paths[:,:,2]
        sinp = torch.sin(phase)
        cosp = torch.cos(phase)

        aoa_az = paths[:, :, 3]
        sin_az = torch.sin(aoa_az)
        cos_az = torch.cos(aoa_az)

        aoa_el = paths[:, :, 4]
        sin_el = torch.sin(aoa_el)
        cos_el = torch.cos(aoa_el)

        x = torch.stack([paths[:,:,0], paths[:,:,1], sinp, cosp, sin_az, cos_az, sin_el,cos_el ], dim=-1)


        interactions_clean = interactions.clone()
        interactions_clean[interactions_clean == -1] = 0

        x = torch.cat([x, interactions_clean], dim=-1)
        # Embed tokens
        x = self.path_in(x)   # (B, T, hidden)
         ## Append SOS embedding
        # x = torch.cat([env_embedding, env_prop_embedding, x], dim=1)

        total_len =   T
        pos = self.pos_emb(torch.arange(total_len, device=x.device))  # (T, hidden)

        x = x + pos
        # Conditioning prompt token

        

        # Construct causal mask
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=x.device), diagonal=1).bool()

        # Decode
        h = self.decoder(
            tgt=x,
            memory=prefix,
            tgt_mask=causal_mask
        )

        h_paths = h[:, :, :]

        # Predict next-step parameters
        out = self.out(h_paths)  # (B, T, 5)

        delay_pred = out[:, :, 0]
        power_pred = out[:, :, 1]
        phase_sin_pred = out[:, :, 2]
        phase_cos_pred = out[:, :, 3]
        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)

        az_sin_pred = out[:, :, 4]
        az_cos_pred = out[:, :, 5]
        el_sin_pred = out[:, :, 6]
        el_cos_pred = out[:, :, 7]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.interaction_head(h_paths)
        ## path count prediction
        prefix_flat = prefix.reshape(B, -1)
        pathcounts = self.pathcount_head(prefix_flat)

        return (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
            az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
            pathcounts, interaction_logits)