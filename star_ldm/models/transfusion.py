import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from functools import partial
from tqdm import tqdm
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional
import math
import os
from omegaconf import DictConfig, OmegaConf, open_dict

from star_ldm.models.modules.diffusion import SinusoidalPosEmb
from star_ldm.models.modules.transformer import TransformerModel
from star_ldm.models.modules.norm import RMSNorm

from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule, log_snr_to_alpha2, alpha2_to_shifted_log_snr
from star_ldm.diffusion.time_sampler import LossEMASampler
from star_ldm.diffusion.diff_utils import predict_noise_from_v, predict_start_from_v, predict_v_from_start_and_eps, predict_noise_from_start, predict_start_from_noise
from star_ldm.diffusion.loss_weighting import get_loss_weighting

from star_ldm.data.CONSTANTS import DATA_STATS_PATH

ModelPrediction =  namedtuple('ModelPrediction', ['pred_eps', 'pred_x', 'pred_v'])


@dataclass
class SampleOutput:
    x_start: torch.Tensor
    x_start_internal: torch.Tensor
    generations: List[str]
    num_branches: int
    num_prompts: int
    selector_scores: Optional[torch.Tensor] = None
    selected_branch: Optional[torch.Tensor] = None
    pairwise_cosine: Optional[torch.Tensor] = None

    def __iter__(self):
        # Compatibility for old call sites that unpacked ``sample`` as
        # ``x_start, generations = model.sample(...)``.
        yield self.x_start
        yield self.generations

def exists(val):
    return val is not None

@torch.amp.autocast('cuda',enabled=False)
def variance_preserving_map(x, alpha2, eps=None):
    if eps is None:
        eps = torch.randn_like(x)

    return alpha2.sqrt() * x + torch.sqrt(1-alpha2) * eps

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


def time_in_guidance_window(time, start, end):
    if time is None:
        return True
    if torch.is_tensor(time):
        time_value = float(time.flatten()[0].detach().item())
    else:
        time_value = float(time)
    elapsed_fraction = 1.0 - time_value
    return start <= elapsed_fraction <= end


def selector_redundancy_loss(
    z_t,
    num_prompts,
    num_branches,
    metric='cosine',
    quality_weights=None,
    tau=1.0,
):
    if num_branches <= 1:
        return z_t.new_zeros(())

    z_bkd = z_t.view(num_prompts, num_branches, z_t.shape[-1])
    upper_mask = torch.triu(
        torch.ones((num_branches, num_branches), dtype=torch.bool, device=z_t.device),
        diagonal=1,
    )

    if metric == 'cosine':
        z_norm = F.normalize(z_bkd, dim=-1)
        pairwise = torch.einsum('bkd,bjd->bkj', z_norm, z_norm)
    elif metric == 'l2':
        d2 = ((z_bkd.unsqueeze(2) - z_bkd.unsqueeze(1)) ** 2).sum(dim=-1)
        pairwise = torch.exp(-d2 / tau)
    else:
        raise ValueError(f'Unknown repulsion metric {metric!r}')

    if quality_weights is not None:
        weights = quality_weights.view(num_prompts, num_branches).to(pairwise)
        pairwise = pairwise * weights.unsqueeze(2) * weights.unsqueeze(1)

    return pairwise[:, upper_mask].sum()


def pairwise_branch_cosine(z_t, num_prompts, num_branches):
    if num_branches <= 1:
        return None
    z_bkd = z_t.view(num_prompts, num_branches, z_t.shape[-1])
    z_norm = F.normalize(z_bkd, dim=-1)
    return torch.einsum('bkd,bjd->bkj', z_norm, z_norm)

class SoftPromptGenerator(nn.Module):
    def __init__(self,
                 sentence_emb_dim=768,
                 transformer_dim=768,
                 prompt_length=8,
                 n_layers=6,
                 dropout=0.0,
                 lm_embed_dim=1280):
        super(SoftPromptGenerator, self).__init__()
        self.splicer = nn.Sequential(
            nn.Linear(sentence_emb_dim, sentence_emb_dim*4),
            Rearrange('b (l d) -> b l d', l=prompt_length),
            nn.Linear(sentence_emb_dim*4//prompt_length, transformer_dim),
        )

        time_emb_dim = sentence_emb_dim//2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sentence_emb_dim),
            nn.Linear(sentence_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.transformer = TransformerModel(
            dim=transformer_dim, num_layers=n_layers, causal=False, pos_emb='absolute', time_emb_dim=time_emb_dim, ff_dropout=dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(transformer_dim, lm_embed_dim),
        )

    def forward(self, noised_sentence_emb, alpha2):
        assert alpha2 is not None
        alpha2 = rearrange(alpha2, 'b ()-> b')
        time_emb = self.time_mlp(alpha2*1000)

        prompt = self.splicer(noised_sentence_emb)
        prompt = self.transformer(prompt, time_emb=time_emb)
        prompt = self.output_proj(prompt)
        return prompt, time_emb

class ScoreNetHead(nn.Module):
    def __init__(self,
                 sentence_emb_dim=768,
                 transformer_dim=768,
                 prompt_length=8,
                 n_layers=4,
                 dropout=0.0,
                 output_dim_mult=4,
                 lm_embed_dim=1280):
        super(ScoreNetHead, self).__init__()
        self.input_proj = nn.Linear(lm_embed_dim*2, transformer_dim)

        time_emb_dim = sentence_emb_dim//2

        self.transformer = TransformerModel(
            dim=transformer_dim, num_layers=n_layers, causal=False, pos_emb='absolute', time_emb_dim=time_emb_dim, ff_dropout=dropout)

        self.output_linear = nn.Sequential(
                nn.Linear(transformer_dim, sentence_emb_dim*output_dim_mult//prompt_length),
                Rearrange('b l d -> b (l d)'),
                nn.Linear(sentence_emb_dim*output_dim_mult, sentence_emb_dim),
            )

    def forward(self, processed_soft_prompt, time_emb):
        assert time_emb is not None

        prompt = self.input_proj(processed_soft_prompt)
        prompt = self.transformer(prompt, time_emb=time_emb)
        prompt = self.output_linear(prompt)
        return prompt


class TransfusionGPT(nn.Module):
    def __init__(self,
                 dataset_name='fineweb_100b',
                 gpt2_model_name='gpt2-large',
                 sentence_encoder_name='sentence-transformers/sentence-t5-xl',
                 transfusion_cfg=None,
                 gamma_min=-15,
                 gamma_max=15,
                 clf_guidance_dropout=0.1,
                 scale_by_std=True,
                 global_norm=False):
        super(TransfusionGPT, self).__init__()
        self.gpt2_model_name = gpt2_model_name
        self.freeze_gpt = transfusion_cfg.train.freeze_gpt
        if transfusion_cfg.train.freeze_gpt:
            self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
            for param in self.gpt2.parameters():
                param.requires_grad = False
        else:
            self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_diffusion_tokens = transfusion_cfg.prompt_generator.prompt_length
        self.model_config = AutoConfig.from_pretrained(gpt2_model_name)

        base_model = self.gpt2
        if 'gpt2' in gpt2_model_name:
            lm_embed_dim = self.model_config.n_embd
            self.lm_embedding = base_model.transformer.wte
        elif 'Llama' in gpt2_model_name:
            lm_embed_dim = self.model_config.hidden_size
            self.lm_embedding = base_model.model.embed_tokens

        # FP16 precision for sentence encoder
        self.sentence_encoder = SentenceTransformer(
            sentence_encoder_name).half()
        # Freeze sentence encoder
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        # Prompt Generator
        self.soft_prompt_generator = SoftPromptGenerator(
            transformer_dim=transfusion_cfg.prompt_generator.dim,
            prompt_length=transfusion_cfg.prompt_generator.prompt_length,
            n_layers=transfusion_cfg.prompt_generator.depth,
            dropout=transfusion_cfg.prompt_generator.dropout,
            lm_embed_dim=lm_embed_dim
        )

        self.null_soft_prompt = nn.Parameter(torch.randn(transfusion_cfg.prompt_generator.prompt_length, lm_embed_dim)*0.02)
        self.clf_guidance_dropout = torch.distributions.Bernoulli(probs=clf_guidance_dropout)

        self.sample_noise_schedule = get_scaled_noise_schedule(
            transfusion_cfg.sampling.noise_schedule_name, scale=transfusion_cfg.sampling.noise_schedule_scale)

        # Diffusion Network
        self.score_net_head = ScoreNetHead(
            transformer_dim=transfusion_cfg.scorenet_head.dim,
            prompt_length=transfusion_cfg.prompt_generator.prompt_length,
            n_layers=transfusion_cfg.scorenet_head.depth,
            dropout=transfusion_cfg.scorenet_head.dropout,
            output_dim_mult=transfusion_cfg.scorenet_head.output_dim_mult,
            lm_embed_dim=lm_embed_dim,
        )

        # Optionally rescale data to have unit variance
        self.scale_by_std = scale_by_std
        if global_norm:
            self.register_buffer('data_mean', torch.load(os.path.join(DATA_STATS_PATH[dataset_name], 'global_mean.pt'), weights_only=True))
            self.register_buffer('data_std', torch.load(os.path.join(DATA_STATS_PATH[dataset_name], 'global_std.pt'), weights_only=True))
        else:
            self.register_buffer('data_mean', torch.load(os.path.join(DATA_STATS_PATH[dataset_name], 'mean.pt'), weights_only=True))
            self.register_buffer('data_std', torch.load(os.path.join(DATA_STATS_PATH[dataset_name], 'std.pt'), weights_only=True))
        self.adaptive_sampler = LossEMASampler(
            n_bins=100, ema_decay=0.9, gamma_min=gamma_min, gamma_max=gamma_max, train_schedule=transfusion_cfg.diffusion_loss.train_schedule, cosine_shift=transfusion_cfg.diffusion_loss.cosine_shift)
        self.train_schedule = transfusion_cfg.diffusion_loss.train_schedule
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.diffusion_loss_weighting = get_loss_weighting(transfusion_cfg.diffusion_loss.weighting_name, **transfusion_cfg.diffusion_loss.weighting_kwargs)

        self.clf_guidance_dropout = torch.distributions.Bernoulli(probs=clf_guidance_dropout)

    def normalize_sentence_emb(self, sentence_emb):
        return (sentence_emb - self.data_mean)/self.data_std

    def unnormalize_sentence_emb(self, sentence_emb):
        return sentence_emb*self.data_std + self.data_mean

    def get_endpoints(self):
        return self.gamma_min, self.gamma_max

    def get_loss_emas(self):
        return self.adaptive_sampler.get_loss_emas()

    def get_unweighted_loss_emas(self):
        return self.adaptive_sampler.get_unweighted_loss_emas()

    def get_weighted_loss(self):
        return self.adaptive_sampler.weights().mean()

    def get_normalized_loss_emas(self):
        return self.adaptive_sampler.get_normalized_loss_emas()

    def get_cdf(self):
        return self.adaptive_sampler.get_cdf()

    def get_sampling_timesteps(self, batch, sampling_timesteps, *, device, start_time=1.0):
        times = torch.linspace(start_time, 0., sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    @torch.amp.autocast('cuda', enabled=False)
    def sample(
        self,
        input_ids,
        diffusion_token_mask=None,
        continuation_start=None,
        sampler='ddpm',
        var_lambda=0.2,
        sampling_timesteps=250,
        cls_free_guidance=1.0,
        sigma2=0.05,
        cosine_scale=3.0,
        cls_guidance=0.0,
        classifier=None,
        cls_target=None,
        generate_kwargs=None,
        selector=None,
        prompt_embeds=None,
        selector_kwargs=None,
        num_plan_branches=1,
        return_internal_latents=False,
        select_best_plan=False,
    ):
        del return_internal_latents
        original_batch = input_ids.shape[0]
        device = input_ids.device
        assert sampler in {'ddim', 'ddpm'}
        assert var_lambda >= 0 and var_lambda <= 1.0
        assert num_plan_branches >= 1

        selector_kwargs = selector_kwargs or {}
        select_best_plan = selector_kwargs.get('select_best_plan', select_best_plan)

        if generate_kwargs is None or not generate_kwargs:
            generate_kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": 32,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }

        if num_plan_branches > 1:
            input_ids = input_ids.repeat_interleave(num_plan_branches, dim=0)
            if diffusion_token_mask is not None:
                diffusion_token_mask = diffusion_token_mask.repeat_interleave(num_plan_branches, dim=0)
            if continuation_start is not None:
                continuation_start = continuation_start.repeat_interleave(num_plan_branches, dim=0)

        batch = input_ids.shape[0]

        if selector is not None:
            if prompt_embeds is None:
                raise ValueError('prompt_embeds must be passed when selector is used')
            prompt_embeds = prompt_embeds.to(device).float()
            if prompt_embeds.ndim == 1:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            if prompt_embeds.shape[0] == original_batch and num_plan_branches > 1:
                prompt_embeds = prompt_embeds.repeat_interleave(num_plan_branches, dim=0)
            if prompt_embeds.shape[0] != batch:
                raise ValueError(
                    f'prompt_embeds batch size must be {batch}, got {prompt_embeds.shape[0]}'
                )

        if exists(cosine_scale):
            sample_noise_schedule = get_scaled_noise_schedule('cosine', scale=cosine_scale)
        else:
            sample_noise_schedule = self.sample_noise_schedule

        time_pairs = self.get_sampling_timesteps(batch, sampling_timesteps=sampling_timesteps, device=device)

        z_t = torch.randn((batch, 768), device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=sampling_timesteps):
            alpha2 = sample_noise_schedule(time).unsqueeze(-1)
            alpha2_next = sample_noise_schedule(time_next).unsqueeze(-1)

            model_output = self.diffusion_model_predictions(
                z_t,
                alpha2,
                input_ids,
                diffusion_token_mask=diffusion_token_mask,
                cls_free_guidance=cls_free_guidance,
                cls_guidance=cls_guidance,
                classifier=classifier,
                cls_target=cls_target,
                selector=selector,
                prompt_embeds=prompt_embeds,
                selector_guidance=selector_kwargs.get('selector_guidance', 0.0),
                selector_guidance_start=selector_kwargs.get('selector_guidance_start', 0.0),
                selector_guidance_end=selector_kwargs.get('selector_guidance_end', 1.0),
                guidance_clip=selector_kwargs.get('guidance_clip', 1.0),
                normalize_guidance_grad=selector_kwargs.get('normalize_guidance_grad', False),
                num_prompts=original_batch,
                num_plan_branches=num_plan_branches,
                repulsion_scale=selector_kwargs.get('repulsion_scale', 0.0),
                repulsion_metric=selector_kwargs.get('repulsion_metric', 'cosine'),
                repulsion_start=selector_kwargs.get('repulsion_start', 0.0),
                repulsion_end=selector_kwargs.get('repulsion_end', 0.7),
                quality_weighted_repulsion=selector_kwargs.get('quality_weighted_repulsion', False),
                repulsion_tau=selector_kwargs.get('repulsion_tau', 1.0),
                current_time=time,
            )

            x_start = model_output.pred_x
            eps = model_output.pred_eps

            if time_next[0] <= 0:
                z_t = x_start
                continue

            if sampler == 'ddim':
                z_t = x_start * alpha2_next.sqrt() + eps * (1-alpha2_next).sqrt()
            elif sampler == 'ddpm':
                noise = torch.randn_like(z_t)
                alpha2_now = alpha2 / alpha2_next

                min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
                max_var = (1.0 - alpha2_now)
                sigma = torch.exp(var_lambda * torch.log(max_var) + (1 - var_lambda) * torch.log(min_var))
                z_t = 1 / alpha2_now.sqrt() * (
                    z_t - (1-alpha2_now) / (1-alpha2).sqrt() * eps
                ) + torch.sqrt(sigma) * noise

        x_start_internal_all = x_start.detach().clone()
        selector_scores = None
        selected_branch = None
        pairwise_cosine = None
        decode_indices = None

        if selector is not None:
            clean_alpha2 = torch.ones((batch, 1), device=device)
            selector_scores = selector.get_logits(
                x_start_internal_all,
                clean_alpha2,
                prompt_embeds,
            ).view(original_batch, num_plan_branches).detach()
            pairwise_cosine = pairwise_branch_cosine(
                x_start_internal_all,
                original_batch,
                num_plan_branches,
            )
            if pairwise_cosine is not None:
                pairwise_cosine = pairwise_cosine.detach()

            if select_best_plan:
                selected_branch = selector_scores.argmax(dim=-1)
                prompt_offsets = torch.arange(original_batch, device=device) * num_plan_branches
                decode_indices = prompt_offsets + selected_branch
        elif select_best_plan and num_plan_branches > 1:
            selected_branch = torch.zeros(original_batch, dtype=torch.long, device=device)
            decode_indices = torch.arange(original_batch, device=device) * num_plan_branches

        x_start_for_decode = x_start_internal_all
        input_ids_for_decode = input_ids
        diffusion_token_mask_for_decode = diffusion_token_mask
        continuation_start_for_decode = continuation_start
        if decode_indices is not None:
            x_start_for_decode = x_start_for_decode.index_select(0, decode_indices)
            input_ids_for_decode = input_ids_for_decode.index_select(0, decode_indices)
            if diffusion_token_mask_for_decode is not None:
                diffusion_token_mask_for_decode = diffusion_token_mask_for_decode.index_select(0, decode_indices)
            if continuation_start_for_decode is not None:
                continuation_start_for_decode = continuation_start_for_decode.index_select(0, decode_indices)

        decode_batch = x_start_for_decode.shape[0]
        alpha2 = torch.full((decode_batch, 1), 1-sigma2, device=device)
        noised_sentence_emb = variance_preserving_map(x_start_for_decode, alpha2)
        soft_prompt, time_emb = self.soft_prompt_generator(
            noised_sentence_emb, alpha2)

        input_embed = self.lm_embedding(input_ids_for_decode).float()
        if diffusion_token_mask_for_decode is not None:
            input_embed[diffusion_token_mask_for_decode] = rearrange(
                    soft_prompt, 'b l d -> (b l) d')
        else:
            input_embed = torch.cat((input_embed, soft_prompt), dim=1)

        gen_id_list = []
        for idx in range(input_embed.shape[0]):
            if diffusion_token_mask_for_decode is not None:
                assert continuation_start_for_decode is not None
                last_diffusion_token = continuation_start_for_decode[idx] + self.num_diffusion_tokens
                idx_input_embed = input_embed[idx:idx+1, :last_diffusion_token]
            else:
                idx_input_embed = input_embed[idx:idx+1]
            if self.freeze_gpt:
                idx_input_embed = idx_input_embed.bfloat16()
            gen_id_list.append(self.gpt2.generate(inputs_embeds=idx_input_embed, **generate_kwargs)[0].tolist())
        generations = self.tokenizer.batch_decode(gen_id_list, skip_special_tokens=True)

        x_start_return = x_start_for_decode
        if self.scale_by_std:
            x_start_return = self.unnormalize_sentence_emb(x_start_return)
            x_start_return = F.normalize(x_start_return, p=2, dim=-1)

        return SampleOutput(
            x_start=x_start_return,
            x_start_internal=x_start_for_decode.detach().clone(),
            generations=generations,
            num_branches=num_plan_branches,
            num_prompts=original_batch,
            selector_scores=selector_scores,
            selected_branch=selected_branch.detach() if selected_branch is not None else None,
            pairwise_cosine=pairwise_cosine,
        )

    def v_pred(self, noised_sentence_emb, input_ids, alpha2, diffusion_token_mask, labels=None, drop_cond=False):
        n_batch = input_ids.shape[0]

        # Get input embeddings
        input_embed = self.lm_embedding(input_ids).float()

        # Generate soft prompt
        soft_prompt, time_emb = self.soft_prompt_generator(
            noised_sentence_emb, alpha2)

        soft_prompt = soft_prompt.float()
        time_emb = time_emb.float()

        # Apply soft prompt
        input_embed[diffusion_token_mask] = rearrange(
            soft_prompt, 'b l d -> (b l) d')

        if drop_cond:
            # For unconditional generation
            diffusion_tokens = self.null_soft_prompt.expand(n_batch, -1, -1)
            ce_loss = None
        else:
            # For conditional generation, call the language model
            gpt2_outputs = self.gpt2(
                inputs_embeds=input_embed.bfloat16(),
                labels=labels,
                output_hidden_states=True,
            )
            ce_loss = gpt2_outputs.loss

            # Extract diffusion tokens from final hidden state
            diffusion_tokens = rearrange(gpt2_outputs.hidden_states[-1][diffusion_token_mask], '(b l) d-> b l d', b=soft_prompt.shape[0], l=soft_prompt.shape[1])

            # Cfg dropout of diffusion tokens, replace batches with null soft prompt
            if self.training:
                drop_mask = self.clf_guidance_dropout.sample((n_batch, 1, 1)).to(diffusion_tokens.device)
                diffusion_tokens = diffusion_tokens*(1-drop_mask) + self.null_soft_prompt*drop_mask

        # Concatenate diffusion tokens with prompt tokens along feature dimension
        diffusion_tokens = torch.cat((soft_prompt, diffusion_tokens), dim=-1)

        # Get score net output
        model_output = self.score_net_head(diffusion_tokens, time_emb)
        v_pred = model_output

        return ce_loss, v_pred

    def forward(self, input_ids, labels, continuation_text, diffusion_token_mask, continuation_emb=None, alpha2=None):
        n_batch = input_ids.shape[0]

        with torch.no_grad():
            assert not (exists(continuation_emb) and exists(continuation_text))
            if exists(continuation_emb):
                sentence_emb = continuation_emb
            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    sentence_emb = self.sentence_encoder.encode(
                        continuation_text, batch_size=n_batch, convert_to_tensor=True, show_progress_bar=False)
            if self.scale_by_std:
                sentence_emb = self.normalize_sentence_emb(sentence_emb)
            else:
                sentence_emb = sentence_emb*math.sqrt(sentence_emb.shape[-1])

            if alpha2 is None:
                gamma, density = self.adaptive_sampler.sample(
                    batch_size=n_batch, device=input_ids.device)
                alpha2 = log_snr_to_alpha2(gamma)
                alpha2 = rearrange(alpha2, 'b -> b ()')
            else:
                density = None
                gamma = alpha2_to_shifted_log_snr(alpha2)
                gamma = gamma.squeeze()

            eps = torch.randn_like(sentence_emb)
            noised_sentence_emb = variance_preserving_map(
                sentence_emb, alpha2, eps=eps)

        ce_loss, v_pred = self.v_pred(
            noised_sentence_emb, input_ids, alpha2, diffusion_token_mask, labels)
        v_target = predict_v_from_start_and_eps(sentence_emb, eps, alpha2)

        unweighted_loss = F.mse_loss(v_pred, v_target, reduction='none')
        unweighted_loss = reduce(unweighted_loss, 'b d -> b', 'mean')

        diffusion_loss_weighting = self.diffusion_loss_weighting.v_loss_weighting(gamma=gamma).squeeze()
        weighted_loss = diffusion_loss_weighting * unweighted_loss
        # Update loss ema
        if self.training:
            with torch.amp.autocast('cuda',enabled=False):
                self.adaptive_sampler.update_ema_buffers(gamma.squeeze(), weighted_loss, unweighted_loss)
        if exists(density):
            # Monte-carlo training loss
            monte_carlo_weighted_loss = torch.exp(torch.log(diffusion_loss_weighting) - torch.log(density))*unweighted_loss
            diffusion_loss = (monte_carlo_weighted_loss).mean()
        else:
            diffusion_loss = weighted_loss.mean()

        # Return loss dict
        loss_dict = {
            'nll_loss': ce_loss,
            'diffusion_loss': diffusion_loss,
            'unweighted_diffusion_loss': unweighted_loss.mean(),
        }

        return loss_dict

    def diffusion_model_predictions(self, z_t, alpha2, input_ids, cls_free_guidance=1.0, diffusion_token_mask=None,
                                    rescale_x=False, cls_guidance=0.0, classifier=None,
                                    cls_target=None, selector=None, prompt_embeds=None,
                                    selector_guidance=0.0, selector_guidance_start=0.0,
                                    selector_guidance_end=1.0, guidance_clip=1.0,
                                    normalize_guidance_grad=False, num_prompts=None,
                                    num_plan_branches=1, repulsion_scale=0.0,
                                    repulsion_metric='cosine', repulsion_start=0.0,
                                    repulsion_end=0.7, quality_weighted_repulsion=False,
                                    repulsion_tau=1.0, current_time=None):
        # Create diffusion token mask
        if diffusion_token_mask is None:
            diffusion_token_mask = torch.zeros((input_ids.shape[0], input_ids.shape[1]+self.num_diffusion_tokens), dtype=torch.bool)
            diffusion_token_mask[:, -self.num_diffusion_tokens:] = True
            input_ids = F.pad(input_ids, (0, self.num_diffusion_tokens), value=self.tokenizer.pad_token_id)
            diffusion_token_mask = diffusion_token_mask.to(input_ids.device)

        _, pred_v = self.v_pred(z_t, input_ids, alpha2, diffusion_token_mask, labels=None, drop_cond=False)

        if cls_free_guidance != 1.0:
            _, unc_pred_v = self.v_pred(z_t, input_ids, alpha2, diffusion_token_mask, labels=None, drop_cond=True)
            # Combine conditional and unconditional predictions
            pred_v = pred_v*cls_free_guidance + unc_pred_v*(1-cls_free_guidance)

        pred_x = predict_start_from_v(z_t, pred_v, alpha2)
        pred_eps = predict_noise_from_v(z_t, pred_v, alpha2)

        if rescale_x:
            assert not self.scale_by_std
            pred_x = F.normalize(pred_x, p=2, dim=-1)*math.sqrt(pred_x.shape[-1])
            pred_eps = predict_noise_from_start(z_t, pred_x, alpha2)
            pred_v = predict_v_from_start_and_eps(pred_x, pred_eps, alpha2)
        else:
            pred_eps = predict_noise_from_v(z_t, pred_v, alpha2)

        guidance_applied = False

        if cls_guidance != 0.0:
            assert exists(classifier)
            sigma2 = 1-alpha2
            with torch.enable_grad():
                z_t.requires_grad = True
                if cls_target == 0.0:
                    target = torch.zeros((pred_x.shape[0], 1), device=pred_x.device)
                elif cls_target == 1.0:
                    target = torch.ones((pred_x.shape[0], 1), device=pred_x.device)
                else:
                    raise ValueError(f'Invalid cls_target {cls_target}')

                cls_loss = classifier.get_loss(z_t, alpha2, target).sum()
                grad = torch.autograd.grad(cls_loss, z_t)[0]
            pred_eps = pred_eps + cls_guidance*sigma2.sqrt()*grad
            pred_x = predict_start_from_noise(z_t, pred_eps, alpha2)
            guidance_applied = True

        if (
            selector is not None
            and selector_guidance != 0.0
            and time_in_guidance_window(current_time, selector_guidance_start, selector_guidance_end)
        ):
            if prompt_embeds is None:
                raise ValueError('prompt_embeds must be passed when selector guidance is enabled')
            if num_prompts is None:
                num_prompts = z_t.shape[0] // num_plan_branches

            sigma2 = 1-alpha2
            with torch.enable_grad():
                guided_z = z_t.detach().requires_grad_(True)
                sel_loss = selector.get_score_loss(
                    guided_z,
                    alpha2,
                    prompt_embeds,
                    target=1.0,
                ).sum()

                red_loss = guided_z.new_zeros(())
                if (
                    repulsion_scale != 0.0
                    and num_plan_branches > 1
                    and time_in_guidance_window(current_time, repulsion_start, repulsion_end)
                ):
                    quality_weights = None
                    if quality_weighted_repulsion:
                        quality_weights = torch.sigmoid(
                            selector.get_logits(guided_z, alpha2, prompt_embeds)
                        ).detach().squeeze(-1)
                    red_loss = selector_redundancy_loss(
                        guided_z,
                        num_prompts=num_prompts,
                        num_branches=num_plan_branches,
                        metric=repulsion_metric,
                        quality_weights=quality_weights,
                        tau=repulsion_tau,
                    )

                total_loss = sel_loss + repulsion_scale * red_loss
                grad = torch.autograd.grad(total_loss, guided_z)[0]

                if normalize_guidance_grad:
                    grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
                if guidance_clip is not None:
                    grad = grad.clamp(-guidance_clip, guidance_clip)

            pred_eps = pred_eps + selector_guidance*sigma2.sqrt()*grad
            pred_x = predict_start_from_noise(z_t, pred_eps, alpha2)
            guidance_applied = True

        return ModelPrediction(pred_eps, pred_x, None if guidance_applied else pred_v)

    @torch.no_grad()
    def get_sentence_embedding(self, sentence):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            sentence_embedding = self.sentence_encoder.encode(
                sentence, batch_size=1, convert_to_tensor=True, show_progress_bar=False)
        if self.scale_by_std:
            sentence_embedding = self.normalize_sentence_emb(sentence_embedding)
        else:
            sentence_embedding = sentence_embedding*math.sqrt(sentence_embedding.shape[-1])
        return sentence_embedding

    @torch.no_grad()
    def get_teacher_forced_logprob(self, teacher_forced_ids, prompt_ids=None, noised_sentence_embedding=None, alpha2=None, return_per_token=False):
        n_batch = teacher_forced_ids.shape[0]
        seq_len = teacher_forced_ids.shape[1]

        # Create initial input embedding
        if prompt_ids is not None:
            input_embed = self.lm_embedding(prompt_ids).float()
        else:
            input_embed = self.lm_embedding(torch.tensor([[self.tokenizer.bos_token_id]], device=teacher_forced_ids.device)).float()

        # Add soft prompt if sentence embedding is provided
        if noised_sentence_embedding is not None:
            assert alpha2 is not None
            soft_prompt, _ = self.soft_prompt_generator(
                noised_sentence_embedding, alpha2)
            input_embed = torch.cat([input_embed, soft_prompt.float()], dim=1)

        # Concatenate with all teacher forced tokens except the last one
        teacher_forced_embed = self.lm_embedding(teacher_forced_ids[:, :-1]).float()
        full_embed = torch.cat([input_embed, teacher_forced_embed], dim=1)
        prefix_len = input_embed.size(1)

        # Single forward pass through GPT2
        outputs = self.gpt2(inputs_embeds=full_embed.bfloat16(),
                            output_hidden_states=False)

        # Get logits at positions where we need to predict the next token
        logits = outputs.logits[:, (prefix_len-1):(prefix_len+seq_len-1), :]

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probabilities for the actual next tokens in the sequence
        token_log_probs = torch.gather(
            log_probs, 2, teacher_forced_ids.unsqueeze(-1)
        ).squeeze(-1)

        if return_per_token:
            return token_log_probs

        # Sum to get sequence log probability
        sequence_log_probs = token_log_probs.sum(dim=1)

        return sequence_log_probs
