import sys
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from einops import rearrange
import math

import numpy as np
from mamba_ssm import Mamba

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )

def lambda_init(lambda_init_value, depth):
    return lambda_init_value - 0.6 * math.exp(-0.3 * (depth - 1))


def get_config_value(config, key, default):
    try:
        return config[key]
    except Exception:
        return default


class ItemToInterestAggregation(nn.Module):
    def __init__(self, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  
        D_matrix = torch.matmul(input_tensor, self.theta)
        D_matrix = F.softmax(D_matrix, dim=-2)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)
        return result



class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.cpu_fallback = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):
        if input_tensor.is_cuda:
            hidden_states = self.mamba(input_tensor)
        else:
            raise RuntimeError(
                "MambaLayer requires CUDA; CPU execution is not supported. "
                "Move the model and inputs to GPU."
            )

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class IntervalRepetitionGate(nn.Module):
    def __init__(
        self,
        num_bins,
        hidden_size,
        log_base=2.0,
        max_days=3650,
        dropout=0.1,
        time_unit_seconds=86400.0,
        use_path_a=True,
        use_path_b=False,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.log_base = max(log_base, 1.0001)
        self.max_days = max_days
        self.time_unit_seconds = time_unit_seconds
        self.use_path_a = use_path_a
        self.use_path_b = use_path_b
        self.interval_embedding = nn.Embedding(num_bins, hidden_size)
        if use_path_a:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        if use_path_b:
            self.anchor_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def _bucketize(self, delta):
        delta = torch.clamp(delta, min=0.0, max=float(self.max_days))
        bucket = torch.zeros_like(delta, dtype=torch.long)
        positive_mask = delta > 0
        if positive_mask.any():
            bucket[positive_mask] = (
                torch.floor(torch.log1p(delta[positive_mask]) / math.log(self.log_base)).long() + 1
            )
        return torch.clamp(bucket, min=0, max=self.num_bins - 1)

    def forward(self, time_seq, anchor_time, seq_mask):
        feat_a, feat_b = None, None
        if self.use_path_a:
            delta = (anchor_time.unsqueeze(1) - time_seq).float() / self.time_unit_seconds
            hidden = self.proj(self.interval_embedding(self._bucketize(delta)))
            feat_a = hidden * seq_mask.unsqueeze(-1).to(hidden.dtype)
        if self.use_path_b:
            masked_time = torch.where(seq_mask, time_seq, torch.zeros_like(time_seq))
            last_time = masked_time.max(dim=1).values
            recency = (anchor_time - last_time).float() / self.time_unit_seconds
            feat_b = self.anchor_proj(self.interval_embedding(self._bucketize(recency)))
        return feat_a, feat_b


class CalendarPeriodicityGate(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout=0.1,
        use_hour=False,
        use_day_of_week=True,
        use_weekend=True,
        use_day_of_month=True,
        use_month=True,
        use_path_a=True,
        use_path_b=True,
        scalar_gate=False,
    ):
        super().__init__()
        self.use_path_a = use_path_a
        self.use_path_b = use_path_b
        self.scalar_gate = scalar_gate
        self.feature_specs = []
        if use_hour:
            self.feature_specs.append(("hour", 24))
        if use_day_of_week:
            self.feature_specs.append(("day_of_week", 7))
        if use_weekend:
            self.feature_specs.append(("weekend", 2))
        if use_day_of_month:
            self.feature_specs.append(("day_of_month", 30))
        if use_month:
            self.feature_specs.append(("month", 12))

        self.hidden_size = hidden_size
        self.anchor_embeddings = nn.ModuleDict(
            {name: nn.Embedding(cardinality, hidden_size) for name, cardinality in self.feature_specs}
        )
        if use_path_a:
            self.embeddings = nn.ModuleDict(
                {name: nn.Embedding(cardinality, hidden_size) for name, cardinality in self.feature_specs}
            )
            match_dim = max(len(self.feature_specs), 1)
            self.match_proj = nn.Linear(match_dim, hidden_size)
            if scalar_gate:
                self.proj = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
            else:
                self.proj = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
        if use_path_b:
            self.anchor_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def _calendar_parts(self, timestamps):
        local_ts = timestamps.float()
        local_day = torch.floor(local_ts / 86400.0).long()
        local_hour = torch.floor(local_ts / 3600.0).long()

        hour = torch.remainder(local_hour, 24)
        day_of_week = torch.remainder(local_day + 3, 7)
        weekend = (day_of_week >= 5).long()
        day_of_month = torch.remainder(local_day, 30)
        month = torch.remainder(torch.floor(local_day.float() / 30.0).long(), 12)
        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "weekend": weekend,
            "day_of_month": day_of_month,
            "month": month,
        }

    def forward(self, time_seq, anchor_time, seq_mask):
        batch_size, seq_len = time_seq.size()
        device = time_seq.device
        feat_a, feat_b = None, None
        if len(self.feature_specs) == 0:
            return feat_a, feat_b

        anchor_parts = self._calendar_parts(anchor_time)
        anchor_feature_embs = {
            name: self.anchor_embeddings[name](anchor_parts[name]) for name, _ in self.feature_specs
        }
        anchor_emb = sum(anchor_feature_embs.values())

        if self.use_path_a:
            seq_parts = self._calendar_parts(time_seq)
            seq_emb = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
            match_values = []
            for name, _ in self.feature_specs:
                seq_feature = seq_parts[name]
                anchor_feature = anchor_parts[name]
                seq_emb = seq_emb + self.embeddings[name](seq_feature)
                seq_emb = seq_emb + anchor_feature_embs[name].unsqueeze(1)
                match_values.append((seq_feature == anchor_feature.unsqueeze(1)).float())
            seq_emb = seq_emb + self.match_proj(torch.stack(match_values, dim=-1))
            hidden = self.proj(seq_emb)
            feat_a = hidden * seq_mask.unsqueeze(-1).to(hidden.dtype)

        if self.use_path_b:
            feat_b = self.anchor_proj(anchor_emb)
        return feat_a, feat_b


class UserTimeGate(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, user_repr, time_hidden):
        interaction = user_repr * time_hidden
        return self.gate(torch.cat([user_repr, time_hidden, interaction], dim=-1))


class ItemTimeGate(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, item_embs, time_hidden):
        interaction = item_embs * time_hidden.unsqueeze(1)
        return torch.sigmoid(self.gate(interaction))


class FrequencyRhythmGate(nn.Module):
    def __init__(
        self,
        hidden_size,
        window_days=64,
        dropout=0.1,
        use_path_a=False,
        use_path_b=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_days = window_days
        self.use_path_a = use_path_a
        self.use_path_b = use_path_b

        n_rfft = window_days // 2 + 1
        self.n_freq = n_rfft - 1
        omega = 2.0 * math.pi * torch.arange(1, n_rfft, dtype=torch.float32) / float(window_days)
        self.register_buffer("omega", omega)

        if use_path_b:
            self.proj_b = nn.Sequential(
                nn.Linear(2 * self.n_freq, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        if use_path_a:
            self.proj_a = nn.Sequential(
                nn.Linear(self.n_freq, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )

    def _build_daily_series(self, time_seq, anchor_time, seq_mask):
        B = time_seq.size(0)
        W = self.window_days
        device = time_seq.device
        anchor_day = torch.floor(anchor_time / 86400.0)
        history_day = torch.floor(time_seq / 86400.0)
        rel_day = (anchor_day.unsqueeze(1) - history_day).long()
        valid = seq_mask & (rel_day >= 0) & (rel_day < W)
        indices = torch.clamp(rel_day, 0, W - 1)
        series = torch.zeros(B, W, device=device)
        series.scatter_add_(1, indices, valid.float())
        return series

    def _amplitude_weights(self, series):
        x = series - series.mean(dim=1, keepdim=True)
        mag = torch.abs(torch.fft.rfft(x, dim=1))
        amp = mag[:, 1:]
        return amp / amp.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, time_seq, anchor_time, seq_mask):
        feat_a, feat_b = None, None
        series = self._build_daily_series(time_seq, anchor_time, seq_mask)
        amp_w = self._amplitude_weights(series)
        W = float(self.window_days)

        if self.use_path_b:
            anchor_day = anchor_time / 86400.0
            phase = self.omega.unsqueeze(0) * torch.remainder(anchor_day, W).unsqueeze(1)
            weighted_sin = amp_w * torch.sin(phase)
            weighted_cos = amp_w * torch.cos(phase)
            feat_b = self.proj_b(torch.cat([weighted_sin, weighted_cos], dim=1))

        if self.use_path_a:
            anchor_day = anchor_time / 86400.0
            history_day = time_seq / 86400.0
            delta_day = anchor_day.unsqueeze(1) - history_day
            delta_phase = self.omega.view(1, 1, -1) * torch.remainder(delta_day, W).unsqueeze(2)
            phase_match = torch.cos(delta_phase)
            weighted_match = amp_w.unsqueeze(1) * phase_match
            feat_a = self.proj_a(weighted_match)
            feat_a = feat_a * seq_mask.unsqueeze(-1).to(feat_a.dtype)

        return feat_a, feat_b


class PTSF(SequentialRecommender):
    def __init__(self, config, dataset):
        super(PTSF, self).__init__(config, dataset)

        
        
        self.margin = config["margin"]
        self.lambda_init_value = config["lambda_init_value"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.seed = config["seed"]

        self.omega1 = config["omega1"] if "omega1" in config else 0.5
        
        
        
        self.bata = config["bata"] if "bata" in config else 0.5
        
        self.ssl_lambda = nn.Parameter(torch.ones(1) * 0.25)
        self.tau = config["tau"] if "tau" in config else 4

        self.hidden_size = config["hidden_size"]
        self.inner_size = config[
            "inner_size"
        ]

        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["hidden_dropout_prob"]

        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.n_facet_all = config["n_facet_all"]
        self.n_facet = config["n_facet"]
        self.n_facet_window = config["n_facet_window"]
        self.n_facet_hidden = min(
            config["n_facet_hidden"], config["n_layers"]
        )
        self.n_facet_MLP = config["n_facet_MLP"]
        self.n_facet_context = config["n_facet_context"]

        self.n_facet_emb = config["n_facet_emb"]
        self.weight_mode = config["weight_mode"]
        self.context_norm = config["context_norm"]
        self.post_remove_context = config["post_remove_context"]
        self.partition_merging_mode = config["partition_merging_mode"]

        if self.weight_mode == "max_logits":
            self.n_facet_effective = 1
        else:
            self.n_facet_effective = self.n_facet

        assert (
            self.n_facet
            + self.n_facet_context
            + self.n_facet_emb
            == self.n_facet_all
        )
        assert self.n_facet_emb == 0 or self.n_facet_emb == 2
        assert self.n_facet_MLP <= 0
        assert self.n_facet_window <= 0
        self.n_facet_window = -self.n_facet_window
        self.n_facet_MLP = -self.n_facet_MLP
        self.softmax_nonlinear = "None"
        self.use_proj_bias = config["use_proj_bias"]
        hidden_state_input_ratio = 1 + self.n_facet_MLP
        self.MLP_linear = nn.Linear(
            self.hidden_size * (self.n_facet_hidden * (self.n_facet_window + 1)),
            self.hidden_size * self.n_facet_MLP,
        )
        total_lin_dim = self.hidden_size * hidden_state_input_ratio
        self.project_arr = nn.ModuleList(
            [
                nn.Linear(total_lin_dim, self.hidden_size, bias=self.use_proj_bias)
                for i in range(self.n_facet_all)
            ]
        )

        self.project_emb = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.use_proj_bias
        )
        if len(self.weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(
                self.hidden_size * hidden_state_input_ratio, self.n_facet_effective
            )
            self.weight_global = nn.Parameter(torch.ones(self.n_facet_effective))
        self.output_probs = True

        
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        
        self.encoder_type = config["encoder_type"]
        if self.encoder_type == "transformer":
            pass
        elif self.encoder_type == "mamba":
            
            print("use mamba")
            self.d_state = config["d_state"]
            self.d_conv = config["d_conv"]
            self.expand = config["expand"]
            
            self.mamba_layers = nn.ModuleList([
                MambaLayer(
                    d_model=self.hidden_size,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.hidden_dropout_prob,
                    num_layers=self.n_layers,
                ) for _ in range(self.n_layers)
            ])
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Must be 'transformer' or 'mamba'.")
        
        self.use_low_rank_attention = config["use_low_rank_attention"]
        self.k_interests = config["k_interests"]
        self.low_rank_mode = config["low_rank_mode"]

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.use_interval_recurrence = bool(int(get_config_value(config, "use_interval_recurrence", 0)))
        self.use_calendar_periodicity = bool(int(get_config_value(config, "use_calendar_periodicity", 0)))
        self.use_frequency_rhythm = bool(int(get_config_value(config, "use_frequency_rhythm", 0)))
        self.interval_use_path_a = bool(int(get_config_value(config, "interval_use_path_a", 1)))
        self.interval_use_path_b = bool(int(get_config_value(config, "interval_use_path_b", 0)))
        self.calendar_use_path_a = bool(int(get_config_value(config, "calendar_use_path_a", 1)))
        self.calendar_use_path_b = bool(int(get_config_value(config, "calendar_use_path_b", 1)))
        self.frequency_use_path_a = bool(int(get_config_value(config, "frequency_use_path_a", 0)))
        self.frequency_use_path_b = bool(int(get_config_value(config, "frequency_use_path_b", 1)))
        self.time_fusion_mode = get_config_value(config, "time_fusion_mode", "concat")
        self.time_gate_shallow = bool(int(get_config_value(config, "time_gate_shallow", 0)))

        interval_pa = self.use_interval_recurrence and self.interval_use_path_a
        interval_pb = self.use_interval_recurrence and self.interval_use_path_b
        calendar_pa = self.use_calendar_periodicity and self.calendar_use_path_a
        calendar_pb = self.use_calendar_periodicity and self.calendar_use_path_b
        frequency_pa = self.use_frequency_rhythm and self.frequency_use_path_a
        frequency_pb = self.use_frequency_rhythm and self.frequency_use_path_b

        self.time_field = config["TIME_FIELD"]
        self.time_seq_field = self.time_field + get_config_value(config, "LIST_SUFFIX", "_list")
        self.time_anchor_mode = get_config_value(config, "time_anchor_mode", "target")
        self.time_gate_alpha = float(get_config_value(config, "time_gate_alpha", 0.5))
        time_gate_dropout = float(get_config_value(config, "time_gate_dropout", 0.1))

        self.interval_repetition_gate = None
        self.calendar_periodicity_gate = None
        self.frequency_rhythm_gate = None
        if interval_pa or interval_pb:
            self.interval_repetition_gate = IntervalRepetitionGate(
                num_bins=int(get_config_value(config, "interval_num_bins", 32)),
                hidden_size=self.hidden_size,
                log_base=float(get_config_value(config, "interval_log_base", 2.0)),
                max_days=float(get_config_value(config, "interval_max_days", 3650)),
                dropout=time_gate_dropout,
                time_unit_seconds=float(get_config_value(config, "time_unit_seconds", 86400.0)),
                use_path_a=interval_pa,
                use_path_b=interval_pb,
            )
        if calendar_pa or calendar_pb:
            self.calendar_periodicity_gate = CalendarPeriodicityGate(
                hidden_size=self.hidden_size,
                dropout=time_gate_dropout,
                use_hour=bool(int(get_config_value(config, "calendar_use_hour", 0))),
                use_day_of_week=bool(int(get_config_value(config, "calendar_use_day_of_week", 1))),
                use_weekend=bool(int(get_config_value(config, "calendar_use_weekend", 1))),
                use_day_of_month=bool(int(get_config_value(config, "calendar_use_day_of_month", 1))),
                use_month=bool(int(get_config_value(config, "calendar_use_month", 1))),
                use_path_a=calendar_pa,
                use_path_b=calendar_pb,
                scalar_gate=self.time_gate_shallow,
            )
        if frequency_pa or frequency_pb:
            self.frequency_rhythm_gate = FrequencyRhythmGate(
                hidden_size=self.hidden_size,
                window_days=int(get_config_value(config, "frequency_window_days", 64)),
                dropout=time_gate_dropout,
                use_path_a=frequency_pa,
                use_path_b=frequency_pb,
            )

        self.path_a_modules = []
        if interval_pa:
            self.path_a_modules.append("interval")
        if calendar_pa:
            self.path_a_modules.append("calendar")
        if frequency_pa:
            self.path_a_modules.append("frequency")
        self.path_b_active = interval_pb or calendar_pb or frequency_pb
        self.time_modules_enabled = bool(self.path_a_modules) or self.path_b_active

        if self.time_gate_shallow and self.path_a_modules != ["calendar"]:
            raise ValueError(
                f"time_gate_shallow=1 仅支持 calendar 单独走路径A,当前 path_a_modules={self.path_a_modules}"
            )

        self.time_fusion = None
        self.time_fusion_heads = None
        self.time_router = None
        k_a = len(self.path_a_modules)
        if k_a > 0 and not self.time_gate_shallow:
            feat_dim = self.hidden_size * k_a
            if self.time_fusion_mode == "router":
                self.time_fusion_heads = nn.ModuleList(
                    [nn.Linear(self.hidden_size, 1) for _ in range(k_a)]
                )
                self.time_router = nn.Linear(feat_dim, k_a)
            else:
                self.time_fusion = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim),
                    nn.GELU(),
                    nn.Dropout(time_gate_dropout),
                    nn.Linear(feat_dim, 1),
                )

        self.user_time_gate = None
        self.item_time_gate = None
        if bool(int(get_config_value(config, "use_user_time_interaction", 0))) and self.path_b_active:
            self.user_time_gate = UserTimeGate(self.hidden_size, dropout=time_gate_dropout)
        if bool(int(get_config_value(config, "use_item_time_interaction", 0))) and self.path_b_active:
            self.item_time_gate = ItemTimeGate(self.hidden_size, dropout=time_gate_dropout)


        if self.loss_type == "BPR":
            print("current softmax-cpr code does not support BPR loss")
            sys.exit(0)
        elif self.loss_type == "CE":
            self.loss_fct = nn.NLLLoss(
                reduction="none", ignore_index=0
            )
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.apply(self._init_weights)
        
        
        embedding_init_method = config["embedding_init_method"]
        
        
        svd_n_iter = config["svd_n_iter"]


        interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        import time
        time_start = time.time()
        
        if embedding_init_method == "range_finder":
            print(f"Using Randomized Range Finder initialization method...")
            
            init_from_rrf(self, interaction_matrix, self.hidden_size, 
                          n_iter=svd_n_iter, random_state=self.seed)
        elif embedding_init_method == "scipy_svd":
            print(f"Using Scipy SVD (ARPACK) initialization method...")
            init_from_scipy_svd(self, interaction_matrix, self.hidden_size)        
        else:
            print(f"Warning: Unknown embedding_init_method '{embedding_init_method}'. Using default random initialization.")
        
        time_end = time.time()
        print(f"Embedding initialization completed in {time_end - time_start:.2f} seconds.")


    def get_facet_emb(self, input_emb, i):
        return self.project_arr[i](input_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            pass
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, user_id, item_seq_len, ):
        
        
        
        
        

        
        
        
        
        input_emb = self.item_embedding(item_seq)
        input_emb = self.dropout(input_emb)

        if self.encoder_type == "transformer":
            trm_output = self.trm_encoder(input_emb)
        elif self.encoder_type == "mamba":
            
            mamba_output = input_emb
            for i in range(self.n_layers):
                mamba_output = self.mamba_layers[i](mamba_output)
            trm_output = mamba_output
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        return [trm_output]

    def _get_time_inputs(self, interaction, item_seq, item_seq_len):
        if not self.time_modules_enabled:
            return None, None, None
        try:
            time_seq = interaction[self.time_seq_field]
        except Exception:
            return None, None, None

        time_seq = time_seq.to(item_seq.device).float()
        if time_seq.dim() > 2:
            time_seq = time_seq.squeeze(-1)
        seq_mask = (item_seq > 0) & (time_seq > 0)

        anchor_time = None
        if self.time_anchor_mode == "target":
            try:
                anchor_time = interaction[self.time_field].to(item_seq.device).float()
                if anchor_time.dim() > 1:
                    anchor_time = anchor_time.squeeze(-1)
            except Exception:
                anchor_time = None

        if anchor_time is None:
            gather_index = torch.clamp(item_seq_len - 1, min=0).view(-1, 1)
            anchor_time = time_seq.gather(dim=1, index=gather_index).squeeze(1)

        return time_seq, anchor_time, seq_mask

    def _compute_time_outputs(self, time_seq, anchor_time, seq_mask):
        if time_seq is None:
            return None
        feats_a = []
        feats_b = []

        if self.interval_repetition_gate is not None:
            ia, ib = self.interval_repetition_gate(time_seq, anchor_time, seq_mask)
            if ia is not None:
                feats_a.append(ia)
            if ib is not None:
                feats_b.append(ib)

        if self.calendar_periodicity_gate is not None:
            ca, cb = self.calendar_periodicity_gate(time_seq, anchor_time, seq_mask)
            if ca is not None:
                feats_a.append(ca)
            if cb is not None:
                feats_b.append(cb)

        if self.frequency_rhythm_gate is not None:
            fa, fb = self.frequency_rhythm_gate(time_seq, anchor_time, seq_mask)
            if fa is not None:
                feats_a.append(fa)
            if fb is not None:
                feats_b.append(fb)

        if not feats_a and not feats_b:
            return None
        return {"feats_a": feats_a, "feats_b": feats_b, "seq_mask": seq_mask}

    def _apply_time_context_scale(self, context_src, time_outputs):
        if time_outputs is None or not time_outputs["feats_a"]:
            return context_src

        feats_a = time_outputs["feats_a"]
        seq_mask = time_outputs["seq_mask"]
        if self.time_gate_shallow:
            time_delta = feats_a[0].squeeze(-1)
            time_scale = 1 + self.time_gate_alpha * torch.tanh(time_delta)
        elif self.time_fusion_mode == "router" and self.time_router is not None:
            deltas = torch.stack(
                [torch.tanh(head(f).squeeze(-1)) for head, f in zip(self.time_fusion_heads, feats_a)],
                dim=-1,
            )
            weights = torch.softmax(self.time_router(torch.cat(feats_a, dim=-1)), dim=-1)
            time_delta = (weights * deltas).sum(dim=-1)
            time_scale = 1 + self.time_gate_alpha * time_delta
        else:
            time_delta = self.time_fusion(torch.cat(feats_a, dim=-1)).squeeze(-1)
            time_scale = 1 + self.time_gate_alpha * torch.tanh(time_delta)
        time_scale = time_scale.masked_fill(~seq_mask, 0.0)
        return context_src * time_scale.unsqueeze(1)

    def calculate_loss_prob(self, interaction, only_compute_prob=False):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.ITEM_ID]
        pos_users = interaction[self.USER_ID]

        
        
        all_hidden_states = self.forward(item_seq, pos_users, item_seq_len)
        time_seq, anchor_time, seq_mask = self._get_time_inputs(interaction, item_seq, item_seq_len)
        time_outputs = self._compute_time_outputs(time_seq, anchor_time, seq_mask)

        if self.loss_type != "CE":
            print(
                "current softmax-cpr code does not support BPR or the losses other than cross entropy"
            )
            sys.exit(0)
        else:
            test_item_emb = self.item_embedding.weight
            device = all_hidden_states[0].device

            hidden_emb_arr = []
            for i in range(self.n_facet_hidden): 
                hidden_states = all_hidden_states[
                    -(i + 1) 
                ]
                device = hidden_states.device
                hidden_emb_arr.append(hidden_states)
                

            if self.n_facet_MLP > 0: 
                stacked_hidden_emb_raw_arr = torch.cat(
                    hidden_emb_arr, dim=-1
                )
                hidden_emb_MLP = self.MLP_linear(
                    stacked_hidden_emb_raw_arr
                )
                stacked_hidden_emb_arr_raw = torch.cat(
                    [hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1
                )
            else:
                stacked_hidden_emb_arr_raw = hidden_emb_arr[0] 

            stacked_hidden_emb_arr = gather_indexes(stacked_hidden_emb_arr_raw, item_seq_len - 1).unsqueeze(dim=1)
            anchor_hidden = None
            if time_outputs is not None and time_outputs["feats_b"]:
                anchor_hidden = torch.stack(time_outputs["feats_b"], dim=0).sum(dim=0)
                if self.user_time_gate is not None:
                    user_repr = stacked_hidden_emb_arr.squeeze(1)
                    personalized = self.user_time_gate(user_repr, anchor_hidden)
                    stacked_hidden_emb_arr = stacked_hidden_emb_arr + personalized.unsqueeze(1)
                else:
                    stacked_hidden_emb_arr = stacked_hidden_emb_arr + anchor_hidden.unsqueeze(1)
                        
            projected_emb_arr = []
            facet_lm_logits_arr = []
            facet_lm_logits_real_arr = []

            for i in range(self.n_facet): 
                
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr, i
                )
                projected_emb_arr.append(projected_emb) 
                lm_logits = F.linear(projected_emb, self.item_embedding.weight, None)
                facet_lm_logits_arr.append(lm_logits)


            for i in range(self.n_facet_context): 
                
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr,
                    self.n_facet
                    + i,
                )
                projected_emb_arr.append(projected_emb) 

            for i in range(self.n_facet_emb): 
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr_raw,
                    self.n_facet
                    + self.n_facet_context
                    + i,
                )
                projected_emb_arr.append(projected_emb)

            if self.bata != 0 and not only_compute_prob:
                orth_loss = Orth_loss(projected_emb_arr[1], projected_emb_arr[0])
            else:
                orth_loss = 0.0

            for i in range(self.n_facet_context):
                bsz, seq_len_1, hidden_size = projected_emb_arr[i].size()
                bsz, seq_len_2 = item_seq.size()
                
                logit_hidden_context = (
                    projected_emb_arr[
                        self.n_facet
                        + i
                    ]
                    .unsqueeze(dim=2)
                    .expand(-1, -1, seq_len_2, -1)
                    * self.item_embedding.weight[item_seq, :]
                    .unsqueeze(dim=1)
                    .expand(-1, seq_len_1, -1, -1)
                ).sum(dim=-1)
                logit_hidden_pointer = 0
                if self.n_facet_emb == 2:
                    pointer = gather_indexes(projected_emb_arr[-2], item_seq_len - 1)
                    logit_hidden_pointer = (
                        pointer.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, seq_len_1, seq_len_2, -1)
                        * projected_emb_arr[-1]
                        .unsqueeze(dim=1)
                        .expand(-1, seq_len_1, -1, -1)
                    ).sum(dim=-1)
                item_seq_expand = item_seq.unsqueeze(dim=1).expand(-1, seq_len_1, -1)
                only_new_logits = torch.zeros_like(facet_lm_logits_arr[i])
                if self.context_norm:
                    context_src = logit_hidden_context + logit_hidden_pointer
                    context_src = self._apply_time_context_scale(context_src, time_outputs)
                    if self.item_time_gate is not None and anchor_hidden is not None:
                        item_embs = self.item_embedding(item_seq)
                        item_time_rel = self.item_time_gate(item_embs, anchor_hidden)
                        context_src = context_src * item_time_rel.squeeze(-1).unsqueeze(1)
                    only_new_logits.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=context_src,
                    )
                    item_count = torch.zeros_like(only_new_logits) + 1e-15
                    item_count.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    only_new_logits = only_new_logits / item_count


                if self.partition_merging_mode == "replace":
                    facet_lm_logits_arr[i].scatter_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.zeros_like(item_seq_expand).to(
                            dtype=facet_lm_logits_arr[i].dtype
                        ),
                    )
                facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits

            weight = None

            prediction_prob = 0
            copy_loss = 0
            for i in range(self.n_facet_effective):
                facet_lm_logits = facet_lm_logits_arr[i]

                if self.softmax_nonlinear == "sigsoftmax":
                    facet_lm_logits_sig = torch.exp(
                        facet_lm_logits - facet_lm_logits.max(dim=-1, keepdim=True)[0]
                    ) * (1e-20 + torch.sigmoid(facet_lm_logits))
                    facet_lm_logits_softmax = (
                        facet_lm_logits_sig
                        / facet_lm_logits_sig.sum(dim=-1, keepdim=True)
                    )
                elif self.softmax_nonlinear == "None":
                    facet_lm_logits_softmax = facet_lm_logits.softmax(
                        dim=-1
                    )

                if self.weight_mode == "dynamic":
                    prediction_prob += facet_lm_logits_softmax * weight[
                        :, :, i
                    ].unsqueeze(-1)
                elif self.weight_mode == "static":
                    prediction_prob += facet_lm_logits_softmax * weight[i]
                else:
                    prediction_prob += (
                        facet_lm_logits_softmax / self.n_facet_effective
                    )
            
            
            
            if self.omega1 != 0 and not only_compute_prob:
                seq_emb = stacked_hidden_emb_arr.squeeze(dim=1)
                item_emb = self.item_embedding(pos_items)
                neighbor_agg_loss = get_neighbor_aggregate_loss(seq_emb, item_emb, self.tau)
            else:
                neighbor_agg_loss = 0.0

            if not only_compute_prob:
                inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
                pos_items = interaction[self.POS_ITEM_ID]
                loss_raw = self.loss_fct(inp, pos_items.view(-1))
                loss = loss_raw.mean()
            else:
                loss = None
            return loss, copy_loss, orth_loss, neighbor_agg_loss, prediction_prob.squeeze(dim=1)

    def calculate_loss(self, interaction):
        loss, copy_loss, orth_loss, neighbor_agg_loss, prediction_prob = self.calculate_loss_prob(interaction)
        
        return loss + self.bata * orth_loss + self.omega1 * neighbor_agg_loss

    def predict(self, interaction):
        print(
            "Current softmax cpr code does not support negative sampling in an efficient way just like RepeatNet.",
            file=sys.stderr,
        )

        loss, copy_loss, orth_loss, neighbor_agg_loss, prediction_prob = self.calculate_loss_prob(
            interaction, only_compute_prob=True
        )
        if self.post_remove_context:
            item_seq = interaction[self.ITEM_SEQ]
            prediction_prob.scatter_(1, item_seq, 0)
        test_item = interaction[self.ITEM_ID]
        prediction_prob = prediction_prob.unsqueeze(-1)
        scores = self.gather_indexes(prediction_prob, test_item).squeeze(-1)

        return scores

    def full_sort_predict(self, interaction):
        loss, copy_loss, orth_loss, neighbor_agg_loss, prediction_prob = self.calculate_loss_prob(
            interaction, only_compute_prob=True
        )
        if self.post_remove_context:
            item_seq = interaction[self.ITEM_SEQ]
            prediction_prob.scatter_(1, item_seq, 0)
        return prediction_prob


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_embd, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.c_fc(x)
        
        
        
        return x


def Orth_loss(fC, fE):
    
    fC_norm = (fC - fC.mean(dim=-1, keepdim=True)) / (fC.std(dim=-1, keepdim=True) + 1e-8)
    fE_norm = (fE - fE.mean(dim=-1, keepdim=True)) / (fE.std(dim=-1, keepdim=True) + 1e-8)
    
    corr = (fC_norm * fE_norm).sum(dim=-1) / fC.shape[-1]
    orth_loss = torch.mean(torch.square(corr))
    return orth_loss

def init_from_scipy_svd(self, interaction_matrix, embedding_size, n_components=None):
    from scipy.sparse.linalg import svds
    import gc
    
    interaction_matrix = interaction_matrix.astype(np.float32)
    n_users, n_items = interaction_matrix.shape
    
    if n_components is None:
        n_components = embedding_size
        
    print(f"Initializing embeddings using Scipy SVD ({n_users}x{n_items}, n_components={n_components})...")
    
    try:
        k = min(n_components, min(n_users, n_items) - 1)
        U, S, Vt = svds(interaction_matrix, k=k)
        
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]
        
        if k < n_components:
            S_padded = np.zeros(n_components, dtype=np.float32)
            Vt_padded = np.zeros((n_components, n_items), dtype=np.float32)
            
            S_padded[:k] = S
            Vt_padded[:k, :] = Vt
            
            S, Vt = S_padded, Vt_padded
        
        s_sqrt = np.sqrt(np.maximum(S, 0))
        
        del interaction_matrix
        gc.collect()
        
        item_embeddings = (Vt.T * s_sqrt[np.newaxis, :]).astype(np.float32)
        
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8
        item_embeddings = item_embeddings / norms 
        
        self.item_embedding.weight.data[:n_items].copy_(torch.from_numpy(item_embeddings).float())
        
        del U, S, Vt, s_sqrt, item_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Scipy SVD initialization completed successfully")
        
    except Exception as e:
        print(f"Scipy SVD initialization failed: {e}, using random initialization")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pass

def init_from_rrf(self, interaction_matrix, embedding_size, n_iter=3, random_state=42):
    from sklearn.utils.extmath import randomized_range_finder
    import gc

    
    interaction_matrix = interaction_matrix.astype(np.float32)
    n_users, n_items = interaction_matrix.shape
    
    print(f"Initializing item embeddings using Randomized Range Finder ({n_users}x{n_items}, n_iter={n_iter}, random_state={random_state})...")

    try:
        
        
        Q = randomized_range_finder(
            interaction_matrix.T,  
            size=embedding_size,    
            n_iter=n_iter,          
            power_iteration_normalizer='QR',
            random_state=random_state  
        )
        
        
        print(f"Range finder output shape: {Q.shape}")
        
        
        print("Computing item embeddings via projection...")
        
        
        del interaction_matrix
        gc.collect()
        
        item_embeddings = Q.astype(np.float32)
        
        
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8
        item_embeddings = item_embeddings / norms  
       
        
        del Q
        gc.collect()
        
        
        self.item_embedding.weight.data.copy_(torch.from_numpy(item_embeddings).float())
        
        
        del item_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Item embedding initialization completed (user embedding remains random)")
        
    except Exception as e:
        print(f"Randomized range finder failed: {e}, using random initialization")
        
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pass

def get_neighbor_aggregate_loss(seq_embeddings, item_embeddings, tau=0.1):
    
    seq_embeddings = F.normalize(seq_embeddings, dim=-1)

    

    pos_score2 = (seq_embeddings * item_embeddings).sum(dim=-1)  
    pos_score =  torch.exp(pos_score2 / tau)

    
   
    seq_seq_sim = 0 
    
    seq_item_sim = torch.matmul(seq_embeddings, item_embeddings.transpose(0, 1))  


    
    total_score = torch.exp(seq_item_sim / tau).sum(dim=1)

    
    na_loss = -torch.log(pos_score / (total_score + 1e-6))

    return torch.mean(na_loss)

def gather_indexes(output, gather_index):
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)
