"""
Train a supervised STAR-LDM selector from collected latent plans.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
from star_ldm.selector import Selector


try:
    from star_ldm.training.trainer import get_adamw_optimizer, get_linear_wsd_schedule
except ImportError:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
        return AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    def get_linear_wsd_schedule(
        optimizer,
        num_total_steps,
        num_warmup_steps,
        decay_ratio=0.1,
        min_lr_ratio=0,
        last_epoch=-1,
    ):
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            decay_start = int(num_total_steps * (1.0 - decay_ratio))
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / float(max(1, num_total_steps - decay_start))
            return max(min_lr_ratio, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda, last_epoch)


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='Train a STAR-LDM selector', parents=[pre_parser])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--lambda_rank', type=float, default=0.0)
    parser.add_argument('--use_timestep_selector', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mlp_dim', type=int, default=768)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--prompt_emb_dim', type=int, default=768)
    parser.add_argument('--dataset_name', type=str, default='fineweb_100b')
    parser.add_argument('--global_norm', action='store_true', default=True)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--cosine_scale', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true')

    if pre_args.config is not None:
        cfg = OmegaConf.load(pre_args.config)
        parser.set_defaults(**OmegaConf.to_container(cfg, resolve=True))

    args = parser.parse_args(remaining)
    if args.data_path is None:
        parser.error('--data_path is required')
    if args.output_dir is None:
        parser.error('--output_dir is required')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variance_preserving_map_local(x, alpha2, eps=None):
    if eps is None:
        eps = torch.randn_like(x)
    return alpha2.sqrt() * x + torch.sqrt(1-alpha2) * eps


class SelectorDataset(Dataset):
    def __init__(self, data_path, use_timestep_selector=False, cosine_scale=3.0):
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        self.use_timestep_selector = use_timestep_selector
        self.cosine_scale = cosine_scale
        self.schedule = get_scaled_noise_schedule('cosine', scale=cosine_scale)
        self.prompt_ids = data['prompt_ids'].long()
        self.labels = data['labels'].float()

        if use_timestep_selector and data.get('zt') is not None:
            zt = data['zt'].float()
            alpha2 = data['alpha2'].float()
            noisy_per_clean = zt.shape[1]
            self.prompt_embs = data['prompt_embs'].float().repeat_interleave(noisy_per_clean, dim=0)
            self.z = zt.reshape(-1, zt.shape[-1])
            self.alpha2 = alpha2.reshape(-1)
            self.labels = self.labels.repeat_interleave(noisy_per_clean)
            self.prompt_ids = self.prompt_ids.repeat_interleave(noisy_per_clean)
            self.on_the_fly = False
        else:
            self.prompt_embs = data['prompt_embs'].float()
            self.z0 = data['z0_internal'].float()
            self.on_the_fly = use_timestep_selector
            if not use_timestep_selector:
                self.z = self.z0
                self.alpha2 = torch.ones(self.z.shape[0], dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.on_the_fly:
            z0 = self.z0[idx]
            t = torch.rand(())
            alpha2 = self.schedule(t).float()
            z = variance_preserving_map_local(z0, alpha2.view(1))
        else:
            z = self.z[idx]
            alpha2 = self.alpha2[idx]

        return {
            'prompt_id': self.prompt_ids[idx],
            'prompt_emb': self.prompt_embs[idx],
            'z': z,
            'alpha2': alpha2,
            'label': self.labels[idx],
        }


def pairwise_rank_loss(logits, labels, prompt_ids):
    losses = []
    flat_logits = logits.squeeze(-1)
    for prompt_id in torch.unique(prompt_ids):
        mask = prompt_ids == prompt_id
        pos = torch.nonzero(mask & (labels > 0.5), as_tuple=False).flatten()
        neg = torch.nonzero(mask & (labels <= 0.5), as_tuple=False).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.softplus(-(flat_logits[pos[0]] - flat_logits[neg[0]])))
    if not losses:
        return flat_logits.new_zeros(())
    return torch.stack(losses).mean()


def compute_auroc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def evaluate(model, dataloader, device, lambda_rank):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    labels_all = []
    scores_all = []
    correct = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            prompt_emb = batch['prompt_emb'].to(device)
            z = batch['z'].to(device)
            alpha2 = batch['alpha2'].to(device).unsqueeze(-1)
            labels = batch['label'].to(device)
            prompt_ids = batch['prompt_id'].to(device)

            logits = model.get_logits(z, alpha2, prompt_emb)
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
            rank = pairwise_rank_loss(logits, labels, prompt_ids)
            loss = bce + lambda_rank * rank

            probs = torch.sigmoid(logits.squeeze(-1))
            correct += ((probs >= 0.5) == (labels >= 0.5)).sum().item()
            count += labels.numel()
            total_loss += loss.item() * labels.numel()
            total_bce += bce.item() * labels.numel()
            total_rank += rank.item() * labels.numel()
            labels_all.extend(labels.detach().cpu().tolist())
            scores_all.extend(probs.detach().cpu().tolist())

    return {
        'loss': total_loss / max(1, count),
        'bce_loss': total_bce / max(1, count),
        'rank_loss': total_rank / max(1, count),
        'accuracy': correct / max(1, count),
        'auroc': compute_auroc(labels_all, scores_all),
    }


def save_checkpoint(path, model, ema, optimizer, step):
    checkpoint = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'step': step,
    }
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SelectorDataset(
        args.data_path,
        use_timestep_selector=args.use_timestep_selector,
        cosine_scale=args.cosine_scale,
    )

    indices = torch.randperm(len(dataset)).tolist()
    val_size = int(len(indices) * args.val_fraction)
    if val_size > 0 and len(indices) - val_size > 0:
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
    else:
        val_indices = []
        train_indices = indices

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
    ) if val_indices else None

    steps_per_epoch = max(1, math.ceil(len(train_indices) / args.batch_size))
    if args.num_steps is None:
        epochs = args.epochs if args.epochs is not None else 1
        num_steps = epochs * steps_per_epoch
    else:
        num_steps = args.num_steps

    model = Selector(
        sentence_emb_dim=args.embedding_dim,
        prompt_emb_dim=args.prompt_emb_dim,
        mlp_dim=args.mlp_dim,
        mlp_hidden_dim=args.hidden_dim,
        mlp_depth=args.num_layers,
        global_norm=args.global_norm,
        dataset_name=args.dataset_name,
    ).to(device)

    optimizer = get_adamw_optimizer(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_wsd_schedule(
        optimizer,
        num_total_steps=num_steps,
        num_warmup_steps=args.warmup_steps,
    )

    try:
        from ema_pytorch import EMA
        ema = EMA(model, beta=0.999, update_every=10, power=3 / 4, update_after_step=1000)
        ema.to(device)
    except ImportError:
        ema = None

    config = {
        'dataset_name': args.dataset_name,
        'global_norm': args.global_norm,
        'embedding_dim': args.embedding_dim,
        'prompt_emb_dim': args.prompt_emb_dim,
        'selector': {
            'sentence_emb_dim': args.embedding_dim,
            'prompt_emb_dim': args.prompt_emb_dim,
            'mlp_dim': args.mlp_dim,
            'mlp_hidden_dim': args.hidden_dim,
            'mlp_depth': args.num_layers,
        },
        'train': vars(args),
    }
    OmegaConf.save(OmegaConf.create(config), os.path.join(args.output_dir, 'config.yaml'))

    best_metric = -float('inf')
    step = 0
    progress = tqdm(total=num_steps, desc='Training selector')
    while step < num_steps:
        model.train()
        for batch in train_loader:
            prompt_emb = batch['prompt_emb'].to(device)
            z = batch['z'].to(device)
            alpha2 = batch['alpha2'].to(device).unsqueeze(-1)
            labels = batch['label'].to(device)
            prompt_ids = batch['prompt_id'].to(device)

            logits = model.get_logits(z, alpha2, prompt_emb)
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
            rank = pairwise_rank_loss(logits, labels, prompt_ids)
            loss = bce + args.lambda_rank * rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update()

            step += 1
            progress.update(1)
            progress.set_postfix({
                'bce': f'{bce.item():.4f}',
                'rank': f'{rank.item():.4f}',
            })

            if val_loader is not None:
                metrics = evaluate(model, val_loader, device, args.lambda_rank)
                metric = metrics['auroc'] if metrics['auroc'] is not None else metrics['accuracy']
            else:
                metric = -loss.item()

            save_checkpoint(os.path.join(args.output_dir, 'model.pt'), model, ema, optimizer, step)
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(os.path.join(args.output_dir, 'best_model.pt'), model, ema, optimizer, step)

            if step >= num_steps:
                break
    progress.close()


if __name__ == '__main__':
    main()
