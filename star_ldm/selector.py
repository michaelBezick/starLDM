import os

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from star_ldm.data.CONSTANTS import DATA_STATS_PATH
from star_ldm.models.classifier import ConditionableMLP
from star_ldm.models.modules.diffusion import SinusoidalPosEmb


class Selector(nn.Module):
    """
    Prompt-conditioned binary selector for STAR-LDM diffusion latents.

    The selector expects latents in the loop-internal normalized sentence
    embedding space used by :class:`TransfusionGPT`; it does not normalize
    latents in ``forward``.
    """

    def __init__(
        self,
        sentence_emb_dim=768,
        mlp_dim=768,
        mlp_hidden_dim=1536,
        mlp_depth=4,
        prompt_emb_dim=768,
        global_norm=True,
        dataset_name='fineweb_100b',
    ):
        super().__init__()
        self.sentence_emb_dim = sentence_emb_dim
        self.prompt_emb_dim = prompt_emb_dim
        self.dataset_name = dataset_name
        self.global_norm = global_norm

        time_emb_dim = sentence_emb_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sentence_emb_dim),
            nn.Linear(sentence_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.noise_conditioned_mlp = ConditionableMLP(
            input_dim=sentence_emb_dim + prompt_emb_dim,
            mlp_dim=mlp_dim,
            hidden_dim=mlp_hidden_dim,
            n_layers=mlp_depth,
            time_cond_dim=time_emb_dim,
        )

        if global_norm:
            mean_name = 'global_mean.pt'
            std_name = 'global_std.pt'
        else:
            mean_name = 'mean.pt'
            std_name = 'std.pt'

        stats_dir = DATA_STATS_PATH[dataset_name]
        self.register_buffer(
            'data_mean',
            torch.load(os.path.join(stats_dir, mean_name), weights_only=True),
        )
        self.register_buffer(
            'data_std',
            torch.load(os.path.join(stats_dir, std_name), weights_only=True),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def normalize_sentence_emb(self, sentence_emb):
        return (sentence_emb - self.data_mean) / self.data_std

    def _prepare_inputs(self, z_t, alpha2, prompt_embeds):
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if prompt_embeds.ndim == 1:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if alpha2.ndim == 0:
            alpha2 = alpha2.expand(z_t.shape[0]).unsqueeze(-1)
        elif alpha2.ndim == 1:
            alpha2 = alpha2.unsqueeze(-1)

        if prompt_embeds.shape[0] == 1 and z_t.shape[0] != 1:
            prompt_embeds = prompt_embeds.expand(z_t.shape[0], -1)

        if z_t.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(
                f'z_t and prompt_embeds batch sizes must match, got '
                f'{z_t.shape[0]} and {prompt_embeds.shape[0]}'
            )
        if alpha2.shape[0] == 1 and z_t.shape[0] != 1:
            alpha2 = alpha2.expand(z_t.shape[0], -1)
        if alpha2.shape[0] != z_t.shape[0]:
            raise ValueError(
                f'alpha2 and z_t batch sizes must match, got '
                f'{alpha2.shape[0]} and {z_t.shape[0]}'
            )
        return z_t, alpha2, prompt_embeds

    def forward(self, prompt_embeds, z_t, alpha2):
        return self.get_logits(z_t, alpha2, prompt_embeds)

    def get_logits(self, z_t, alpha2, prompt_embeds):
        z_t, alpha2, prompt_embeds = self._prepare_inputs(z_t, alpha2, prompt_embeds)
        alpha2_time = rearrange(alpha2, 'b () -> b')
        time_emb = self.time_mlp(alpha2_time * 1000)
        selector_input = torch.cat((z_t, prompt_embeds), dim=-1)
        return self.noise_conditioned_mlp(selector_input, time_emb)

    def get_score_loss(self, z_t, alpha2, prompt_embeds, target=1.0):
        logits = self.get_logits(z_t, alpha2, prompt_embeds)
        labels = torch.full_like(logits, float(target))
        return self.loss_fn(logits, labels).squeeze(dim=1)

    def get_loss(self, z_t, alpha2, prompt_embeds, labels):
        logits = self.get_logits(z_t, alpha2, prompt_embeds)
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        return self.loss_fn(logits, labels).squeeze(dim=1)


def _get_nested(cfg, path, default):
    value = cfg
    for key in path:
        if value is None or key not in value:
            return default
        value = value[key]
    return value


def _selector_from_config(config_path):
    if not os.path.exists(config_path):
        return Selector()

    cfg = OmegaConf.load(config_path)
    selector_cfg = cfg.get('selector', {})
    mlp_cfg = selector_cfg.get('mlp_arch', selector_cfg)

    return Selector(
        sentence_emb_dim=selector_cfg.get(
            'sentence_emb_dim',
            cfg.get('embedding_dim', _get_nested(cfg, ('mlp', 'embedding_dim'), 768)),
        ),
        prompt_emb_dim=selector_cfg.get('prompt_emb_dim', cfg.get('prompt_emb_dim', 768)),
        mlp_dim=mlp_cfg.get(
            'mlp_dim',
            mlp_cfg.get('dim', _get_nested(cfg, ('mlp', 'mlp_arch', 'dim'), 768)),
        ),
        mlp_hidden_dim=mlp_cfg.get(
            'mlp_hidden_dim',
            mlp_cfg.get(
                'hidden_dim',
                _get_nested(cfg, ('mlp', 'mlp_arch', 'hidden_dim'), 1536),
            ),
        ),
        mlp_depth=mlp_cfg.get(
            'mlp_depth',
            mlp_cfg.get('depth', _get_nested(cfg, ('mlp', 'mlp_arch', 'depth'), 4)),
        ),
        global_norm=cfg.get('global_norm', True),
        dataset_name=cfg.get('dataset_name', 'fineweb_100b'),
    )


def load_selector(checkpoint_path, device='cuda'):
    """
    Load a selector checkpoint from a directory or a direct ``.pt`` file.

    The loader mirrors ``load_classifier``: it prefers ``best_model.pt`` over
    ``model.pt`` in checkpoint directories and uses EMA weights when present.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, 'config.yaml')
        pt_path = os.path.join(checkpoint_path, 'best_model.pt')
        if not os.path.exists(pt_path):
            pt_path = os.path.join(checkpoint_path, 'model.pt')
    else:
        pt_path = checkpoint_path
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.yaml')

    model = _selector_from_config(config_path)
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and checkpoint.get('ema') is not None:
        from ema_pytorch import EMA

        ema = EMA(model, beta=0.999, update_every=10, power=3 / 4, update_after_step=1000)
        ema.load_state_dict(checkpoint['ema'], strict=False)
        model = ema.ema_model
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model
