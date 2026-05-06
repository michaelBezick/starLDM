import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json
from omegaconf import OmegaConf

from star_ldm.models.transfusion import TransfusionGPT, variance_preserving_map
from star_ldm.diffusion.noise_schedule import log_snr_to_alpha2


@dataclass
class GenerateResult:
    decoded: List[List[str]]
    selector_scores: Optional[List[List[float]]] = None
    selected_branch: Optional[List[int]] = None
    pairwise_cosine: Optional[List[List[List[float]]]] = None
    plan_stats_path: Optional[str] = None

    def flatten(self) -> List[str]:
        return [generation for group in self.decoded for generation in group]


class TransfusionGPTInterface:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        classifier_path: Optional[str] = None,
        selector_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to the STAR-LDM checkpoint directory or ``.pt`` file.
            device: Device to load models onto.
            classifier_path: Optional path to a pretrained
                :class:`~star_ldm.models.classifier.NoiseConditionedMLP` checkpoint
                for classifier-guided generation.
            selector_path: Optional path to a selector checkpoint for
                selector-guided latent planning.
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.gpt2.config._name_or_path)

        self.classifier = None
        if classifier_path is not None:
            from star_ldm.models.classifier import load_classifier
            self.classifier = load_classifier(classifier_path, device=str(self.device))

        self.selector = None
        if selector_path is not None:
            from star_ldm.selector import load_selector
            self.selector = load_selector(selector_path, device=str(self.device))

    def _load_model(self, model_path: str) -> 'TransfusionGPT':
        # Check if model_path ends in '.pt'
        if model_path.endswith('.pt'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
            model_path = os.path.join(model_dir, 'model.pt')

        # Grab model directory from model_path
        transfusion_cfg = OmegaConf.load(os.path.join(model_dir, 'args.yaml'))

        model = TransfusionGPT(
            dataset_name=transfusion_cfg.dataset_name,
            transfusion_cfg=transfusion_cfg,
            gpt2_model_name=transfusion_cfg.train.lm_name,
            gamma_min=-15,
            gamma_max=15,
            clf_guidance_dropout=0.1,
            scale_by_std=True,
            global_norm=transfusion_cfg.train.get('global_norm', False),
        )

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'ema' in ckpt:
            # Direct training checkpoint — extract EMA weights
            from ema_pytorch import EMA
            ema = EMA(model, beta=0.999, update_every=10, power=3/4, update_after_step=1000)
            ema.load_state_dict(ckpt['ema'], strict=False)
            model = ema.ema_model.to(self.device)
        else:
            # Plain state_dict
            state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)

        return model

    def generate(self, prompts: List[str], cls_guidance: float = 0.0,
                 cls_target: Optional[float] = None,
                 selector_guidance: float = 0.0,
                 selector_guidance_start: float = 0.0,
                 selector_guidance_end: float = 1.0,
                 guidance_clip: Optional[float] = 1.0,
                 normalize_guidance_grad: bool = False,
                 num_plan_branches: int = 1,
                 repulsion_scale: float = 0.0,
                 repulsion_metric: str = 'cosine',
                 repulsion_start: float = 0.0,
                 repulsion_end: float = 0.7,
                 quality_weighted_repulsion: bool = False,
                 select_best_plan: bool = False,
                 save_plan_stats: Optional[str] = None,
                 **kwargs) -> GenerateResult:
        """
        Generate text for a list of prompts.

        Args:
            prompts: List of prompts to generate from.
            cls_guidance: Classifier guidance scale. ``0.0`` disables guidance.
                Positive values steer toward ``cls_target``.
            cls_target: Target class for classifier guidance (``0.0`` or ``1.0``).
                Required when ``cls_guidance != 0``.
            selector_guidance: Selector guidance scale. ``0.0`` disables
                selector steering even if a selector is loaded.
            num_plan_branches: Number of parallel diffusion branches per prompt.
            select_best_plan: Decode only the highest-scoring branch when a
                selector is loaded.
            save_plan_stats: Optional JSONL path for selector scores and
                branch diversity diagnostics.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`TransfusionGPT.sample`.

        Returns:
            A :class:`GenerateResult` grouped as ``decoded[prompt][branch]``.
        """
        if cls_guidance != 0.0:
            if self.classifier is None:
                raise ValueError(
                    "Classifier guidance requested but no classifier loaded. "
                    "Pass classifier_path when constructing TransfusionGPTInterface."
                )
            if cls_target is None:
                raise ValueError(
                    "cls_target must be specified (0.0 or 1.0) when using classifier guidance."
                )

        selector_requested = (
            selector_guidance != 0.0
            or repulsion_scale != 0.0
            or select_best_plan
        )
        if selector_requested and self.selector is None:
            raise ValueError(
                "Selector guidance, repulsion, or best-plan selection requested "
                "but no selector loaded. Pass selector_path when constructing "
                "TransfusionGPTInterface."
            )
        if repulsion_scale != 0.0 and selector_guidance == 0.0:
            raise ValueError('repulsion_scale requires selector_guidance to be non-zero')
        if repulsion_metric not in {'cosine', 'l2'}:
            raise ValueError("repulsion_metric must be 'cosine' or 'l2'")
        if num_plan_branches < 1:
            raise ValueError('num_plan_branches must be >= 1')

        decoded = []
        selector_scores = [] if self.selector is not None else None
        selected_branch = [] if select_best_plan else None
        pairwise_cosine = [] if self.selector is not None and num_plan_branches > 1 else None
        stats_rows = []

        generate_kwargs = kwargs.pop('generate_kwargs', {})
        selector_kwargs = dict(
            selector_guidance=selector_guidance,
            selector_guidance_start=selector_guidance_start,
            selector_guidance_end=selector_guidance_end,
            guidance_clip=guidance_clip,
            normalize_guidance_grad=normalize_guidance_grad,
            repulsion_scale=repulsion_scale,
            repulsion_metric=repulsion_metric,
            repulsion_start=repulsion_start,
            repulsion_end=repulsion_end,
            quality_weighted_repulsion=quality_weighted_repulsion,
            select_best_plan=select_best_plan,
        )

        for prompt_id, prompt in enumerate(tqdm(prompts, desc="Generating")):
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            prompt_embeds = None
            if self.selector is not None:
                prompt_embeds = self.model.get_sentence_embedding(prompt).to(self.device).float()

            sample_output = self.model.sample(
                input_ids,
                cls_guidance=cls_guidance,
                classifier=self.classifier,
                cls_target=cls_target,
                generate_kwargs=generate_kwargs,
                selector=self.selector,
                prompt_embeds=prompt_embeds,
                selector_kwargs=selector_kwargs,
                num_plan_branches=num_plan_branches,
                select_best_plan=select_best_plan,
                **kwargs,
            )

            if select_best_plan:
                prompt_decoded = [sample_output.generations[0]]
            else:
                prompt_decoded = sample_output.generations
            decoded.append(prompt_decoded)

            all_scores = None
            if sample_output.selector_scores is not None:
                all_scores = sample_output.selector_scores[0].detach().cpu().tolist()
                if select_best_plan:
                    branch_idx = int(sample_output.selected_branch[0].detach().cpu().item())
                    selector_scores.append([float(all_scores[branch_idx])])
                else:
                    selector_scores.append([float(score) for score in all_scores])

            prompt_pairwise = None
            if sample_output.pairwise_cosine is not None:
                prompt_pairwise = sample_output.pairwise_cosine[0].detach().cpu().tolist()
                pairwise_cosine.append(prompt_pairwise)

            selected_idx = None
            if sample_output.selected_branch is not None:
                selected_idx = int(sample_output.selected_branch[0].detach().cpu().item())
                if selected_branch is not None:
                    selected_branch.append(selected_idx)

            if save_plan_stats is not None:
                stats_rows.append({
                    'prompt_id': prompt_id,
                    'prompt': prompt,
                    'selector_scores': all_scores,
                    'pairwise_cosine': prompt_pairwise,
                    'selected_branch': selected_idx,
                    'decoded': prompt_decoded,
                    'correct': None,
                })

        if save_plan_stats is not None:
            stats_dir = os.path.dirname(save_plan_stats)
            if stats_dir:
                os.makedirs(stats_dir, exist_ok=True)
            with open(save_plan_stats, 'w') as f:
                for row in stats_rows:
                    f.write(json.dumps(row) + '\n')

        return GenerateResult(
            decoded=decoded,
            selector_scores=selector_scores,
            selected_branch=selected_branch,
            pairwise_cosine=pairwise_cosine,
            plan_stats_path=save_plan_stats,
        )

    def generate_flat(self, prompts: List[str], **kwargs) -> List[str]:
        return self.generate(prompts, **kwargs).flatten()

    def interactive_demo(self, generate_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run an interactive demo allowing the user to try different generation settings.
        """
        print("STAR-LDM Interactive Demo")
        print("Enter 'quit' to exit")

        while True:
            prompt = input("\nEnter a prompt: ")
            if prompt.lower() == 'quit':
                break

            if generate_kwargs is None:
                generation = self.generate_flat([prompt])[0]
            else:
                generation = self.generate_flat([prompt], **generate_kwargs)[0]

            print(f"Generated text: {generation}")
