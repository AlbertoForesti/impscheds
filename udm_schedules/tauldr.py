# Uniform-rate CTMC used for the piano experiment: the sequence-transformer
# model and the tau-leaping scheduler needed to load and sample the pretrained
# tauLDR piano checkpoint. Adapted from tauLDR
# (https://github.com/andrew-cr/tauLDR, lib/models/models.py) through the
# Jump Your Steps code base (https://github.com/enkeejunior1/jump-your-steps).

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scheduler(nn.Module):
    """Forward process and reverse rates of a tauLDR uniform-rate CTMC."""
    def __init__(self, args, model):
        super().__init__()
        self.num_vocabs = args.num_vocabs
        self.length = args.length
        self.eps = args.eps

        self.sigma_bar = lambda x: x
        self.qt0 = model.transition
        self.rate = model.rate
        # additive floor of the q_t|0 denominator, as tauLDR's sampler.eps_ratio
        self.eps_ratio = getattr(args, "ratio_eps", None) or 1e-9

    def _qt0_denom(self, qt0, xt, B, L, S, device):
        return qt0[
            torch.arange(B, device=device).repeat_interleave(L*S),
            torch.arange(S, device=device).repeat(B*L),
            xt.long().flatten().repeat_interleave(S)
        ].view(B, L, S) + self.eps_ratio

    def get_score(self, output, t=None, xt=None):
        B, L, S = output.shape # |batch_size, length, num_vocabs|
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == B
        if isinstance(t, float):
            t = t * torch.ones(B).to(output)

        qt0 = self.qt0(t)
        qt0_denom = self._qt0_denom(qt0, xt, B, L, S, output.device)
        return (output / qt0_denom) @ qt0

    def output_to_rev_rate(self, output, t=None, xt=None):
        B, L, S = output.shape # |batch_size, length, num_vocabs|
        device = output.device
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == B
        if isinstance(t, float):
            t = t * torch.ones(B).to(output)

        qt0 = self.qt0(t)
        qt0_denom = self._qt0_denom(qt0, xt, B, L, S, device)
        rate = self.rate(t)
        forward_rates = rate[
            torch.arange(B, device=device).repeat_interleave(L*S),
            torch.arange(S, device=device).repeat(B*L),
            xt.long().flatten().repeat_interleave(S)
        ].view(B, L, S)
        score = (output / qt0_denom) @ qt0
        rev_rate = forward_rates * score
        rev_rate[
            torch.arange(B, device=device).repeat_interleave(L),
            torch.arange(L, device=device).repeat(B),
            xt.long().flatten()
        ] = 0.0 # for Poisson sampling
        return rev_rate

    def sample_latent(self, num_samples):
        return torch.randint(low=0, high=self.num_vocabs, size=(num_samples, self.length))

    def add_noise(self, samples: torch.LongTensor, t, generator=None):
        '''x0 -> xt'''
        B, _ = samples.shape
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == B
        if isinstance(t, float):
            t = t * torch.ones(B).to(samples)
        qt0 = self.qt0(t)
        perturbed_samples = F.one_hot(samples, num_classes=self.num_vocabs).float() @ qt0
        return sample_categorical(perturbed_samples, generator=generator)


class EulerScheduler(Scheduler):
    """Tau-leaping step; the last step takes the argmax of the x0 prediction."""
    def step(self, output, xt, t, step_size, rev_rate=None, generator=None, if_last=False):
        if rev_rate is None:
            rev_rate = self.output_to_rev_rate(output, t=t, xt=xt)

        if if_last:
            xt = torch.max(output, dim=2)[1] # argmax
            rev_rate = None
        else:
            diffs = torch.arange(self.num_vocabs, device=xt.device).view(1,1,self.num_vocabs) - xt[..., None]
            jump_nums = torch.distributions.poisson.Poisson(rev_rate * step_size).sample()
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xt = xt + overall_jump
            xt = torch.clamp(xt, min=0, max=self.num_vocabs-1)
        return SchedulerOutput(xt, xt_prob=None, rev_rate=rev_rate)


class SchedulerOutput:
    def __init__(self, xt, xt_prob=None, rev_rate=None):
        self.xt = xt
        self.xt_prob = xt_prob
        self.rev_rate = rev_rate


def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1)


##########
# Models #
##########
class UniformRate():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_const = cfg.model.rate_const
        self.device = device

        rate = self.rate_const * np.ones((S,S))
        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(rate)

        self.rate_matrix = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    def rate(self, t, #["B"]
    ):
        B = t.shape[0]
        S = self.S

        return torch.tile(self.rate_matrix.view(1,S,S), (B, 1, 1))

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S
        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(self.eigvals.view(1, S) * t.view(B,1))) @ \
            self.eigvecs.T.view(1, S, S)

        if torch.min(transitions) < -1e-6:
            print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        transitions[transitions < 1e-8] = 0.0

        return transitions


class SequenceTransformer(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        num_layers = cfg.model.num_layers
        d_model = cfg.model.d_model
        num_heads = cfg.model.num_heads
        dim_feedforward = cfg.model.dim_feedforward
        dropout = cfg.model.dropout
        num_output_FFresiduals = cfg.model.num_output_FFresiduals
        time_scale_factor = cfg.model.time_scale_factor
        temb_dim = cfg.model.temb_dim
        use_one_hot_input = cfg.model.use_one_hot_input
        self.S = cfg.data.S

        assert len(cfg.data.shape) == 1
        max_len = cfg.data.shape[0]

        self.net = TransformerEncoder(
            num_layers, d_model, num_heads, dim_feedforward, dropout,
            num_output_FFresiduals, time_scale_factor, self.S, max_len,
            temb_dim, use_one_hot_input, device
        ).to(device)

        self.data_shape = cfg.data.shape

    def forward(self,
        x, # ["B", "L"]
        t, # ["B"]
        **kwargs,
    ):
        """
            Returns the x0 prediction (probabilities over the state space)
        """
        # NOTE: t.long() truncates t in [0, 1) to 0; kept as in the code that
        # produced the reported piano results.
        logits = self.net(x.long(), t.long()) # (B, L, S)

        return logits.softmax(dim=-1)


# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA():
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [p.clone().detach()
                            for p in self.parameters() if p.requires_grad]

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd['ema_decay'] = self.decay
        sd['ema_num_updates'] = self.num_updates
        sd['ema_shadow_params'] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(self, state_dict, strict=False)

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)

        if not (len(unexpected_keys) == 3 and \
            'ema_decay' in unexpected_keys and \
            'ema_num_updates' in unexpected_keys and \
            'ema_shadow_params' in unexpected_keys):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict['ema_decay']
        self.num_updates = state_dict['ema_num_updates']
        self.shadow_params = state_dict['ema_shadow_params']

    def train(self, mode=True):
        if self.training == mode:
            print("Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters")
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()


# make sure EMA inherited first so it can override the state dict functions
class UniformRateSequenceTransformerEMA(EMA, SequenceTransformer, UniformRate):
    def __init__(self, cfg, device):
        EMA.__init__(self, cfg)
        SequenceTransformer.__init__(self, cfg, device)
        UniformRate.__init__(self, cfg, device)

        self.init_ema()


#Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x, # ["B", "L", "K"]
    ):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
            dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)

    def forward(self,
        x, # ["B", "L", "K"],
        temb, # ["B", "temb_dim"]
    ):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm1(x + self._sa_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + self._ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]

        return x

    def _sa_block(self, x):
        x = self.self_attn(x,x,x)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FFResidual(nn.Module):
    def __init__(self, d_model, hidden, temb_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)

    def forward(self, x, temb):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm(x + self.linear2(self.activation(self.linear1(x))))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,
        dropout, num_output_FFresiduals, time_scale_factor, S, max_len,
        temb_dim, use_one_hot_input, device):
        super().__init__()

        self.temb_dim = temb_dim
        self.use_one_hot_input = use_one_hot_input

        self.S = S

        self.pos_embed = PositionalEncoding(device, d_model, dropout, max_len)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                    dropout, 4*temb_dim)
            )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.output_resid_layers = []
        for i in range(num_output_FFresiduals):
            self.output_resid_layers.append(
                FFResidual(d_model, dim_feedforward, 4*temb_dim)
            )
        self.output_resid_layers = nn.ModuleList(self.output_resid_layers)

        self.output_linear = nn.Linear(d_model, self.S)

        if use_one_hot_input:
            self.input_embedding = nn.Linear(S, d_model)
        else:
            self.input_embedding = nn.Linear(1, d_model)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 4*temb_dim)
        )

        self.time_scale_factor = time_scale_factor

    def forward(self, x, # ["B", "L"],
        t #["B"]
    ):
        B, L = x.shape

        temb = self.temb_net(
            transformer_timestep_embedding(
                t*self.time_scale_factor, self.temb_dim
            )
        )
        one_hot_x = nn.functional.one_hot(x, num_classes=self.S) # (B, L, S)

        if self.use_one_hot_input:
            x = self.input_embedding(one_hot_x.float()) # (B, L, K)
        else:
            x = self.normalize_input(x)
            x = x.view(B, L, 1)
            x = self.input_embedding(x) # (B, L, K)

        x = self.pos_embed(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, temb)

        # x (B, L, K)
        for resid_layer in self.output_resid_layers:
            x = resid_layer(x, temb)

        x = self.output_linear(x) # (B, L, S)

        x = x + one_hot_x

        return x

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x


# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
