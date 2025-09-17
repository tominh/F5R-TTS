import gc
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange
from ema_pytorch import EMA
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import (default, exists, get_tokenizer, mask_from_start_end_indices, load_checkpoint)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from rl import reward


def mask_from_frac_lengths(seq_len, frac_lengths):
    max_start = (frac_lengths * seq_len).long()

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    start = torch.min(start, dim=-1, keepdim=True).values.repeat(start.size(0))
    prompt_idx = mask_from_start_end_indices(seq_len, (0 * start).long(), start)
    trg_idx = mask_from_start_end_indices(seq_len, start, seq_len)

    return prompt_idx, trg_idx


def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    ode_method = "euler"
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, device)

    return model


class GRPOTrainer():

    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs
        )

        if exists(wandb_resume_id):
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, 'id': wandb_resume_id}}
        else:
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
        self.accelerator.init_trackers(
            project_name=wandb_project,
            init_kwargs=init_kwargs,
            config={"epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler}
        )

        self.repeat_count = 8
        self.mini_repeat_count = 1

        self.model = model

        if self.is_main:
            self.ema_model = EMA(
                model,
                include_online_model=False,
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, 'ckpts/test_e2-tts')

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        ema_model = load_model("F5-TTS", "F5TTS_ref", DiT, F5TTS_model_cfg, "last")
        self.ref_model = ema_model
        self.ref_model.eval()
        self.ref_model = self.accelerator.prepare(self.ref_model)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if not exists(self.checkpoint_path):
            return 0
        if not os.path.exists(self.checkpoint_path):
            return 0
        if not os.listdir(self.checkpoint_path):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pt')],
                key=lambda x: int(''.join(filter(str.isdigit, x)))
            )[-1]
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)

        if 'step' in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            step = checkpoint['step']
        else:
            temp = {k.replace("ema_model.", ""): v for k, v in checkpoint['ema_model_state_dict'].items()}
            checkpoint['model_state_dict'] = {k: v for k, v in temp.items() if k not in ["initted", "step"]}
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'], strict=False)
            step = 0

        del checkpoint
        gc.collect()
        return step

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
                persistent_workers=True, batch_size=self.batch_size, shuffle=True, generator=generator
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed,
                drop_last=True, repeat_count=self.repeat_count, mini_repeat_count=self.mini_repeat_count
            )
            train_dataloader = DataLoader(
                train_dataset, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
                persistent_workers=True, batch_sampler=batch_sampler
            )
        else:
            raise ValueError(
                f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
            )

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_step = self.load_checkpoint()
        global_step = start_step

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch, total=orig_epoch_step
                )
            else:
                progress_bar = tqdm(
                    train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="step",
                    disable=not self.accelerator.is_local_main_process
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch['text']
                    mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                    mel_lengths = batch["mel_lengths"]
                    text_len = max([len(item) for item in text_inputs])
                    random_int = random.randint(1, 50) + int(str(mel_spec.device)[-1])
                    torch.manual_seed(random_int)
                    if text_len > max(mel_lengths):
                        continue

                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get('durations'))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_step)

                    frac_lengths = torch.zeros((mel_spec.size(0),), device=self.model.device)
                    frac_lengths = frac_lengths.float().uniform_(*(0.1, 0.3))
                    prompt_idx, trg_idx = mask_from_frac_lengths(mel_lengths, frac_lengths)
                    prompt_idx = prompt_idx.unsqueeze(-1)
                    prompt_idx = prompt_idx.repeat(1, 1, 100)
                    prompt_audio = mel_spec[prompt_idx].view(mel_spec.size(0), -1, mel_spec.size(2))
                    out, _, pro_result = self.model.module.forward_rl(
                        cond=prompt_audio,
                        text=text_inputs,
                        duration=mel_lengths,
                        steps=30,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1.0,
                    )
                    with torch.no_grad():
                        _, _, ref_pro_result = self.ref_model.module.forward_rl(
                            cond=prompt_audio,
                            text=text_inputs,
                            duration=mel_lengths,
                            steps=30,
                            cfg_strength=2.0,
                            sway_sampling_coef=-1.0,
                        )
                    pro_result_sample = []
                    ref_pro_result_sample = []
                    for i, item in enumerate(pro_result):
                        if item[-1]:
                            pro_result_sample.append(item[:-1])
                            ref_pro_result_sample.append(ref_pro_result[i][:-1])
                    pro_result = pro_result_sample
                    ref_pro_result = ref_pro_result_sample
                    sim, acc = reward.get_reward(out, mel_spec)
                    rewards = sim * 1.0 + acc * 3.0

                    # Compute grouped-wise rewards
                    rewards_list = rewards.view(-1).tolist()
                    rewards_list = [str(item) for item in rewards_list]
                    with open("./{}".format(rewards.device), "w") as f:
                        f.write("\n".join(rewards_list))
                    self.accelerator.wait_for_everyone()
                    rewards_list = []
                    for i in range(torch.cuda.device_count()):
                        temp = []
                        with open(f"cuda:{i}") as f:
                            for line in f.readlines():
                                line = line.strip("\n")
                                temp.append(float(line))
                        rewards_list.append(temp)
                    mean = torch.from_numpy(np.mean(rewards_list, axis=0)).to(rewards.device).to(rewards.dtype)
                    std = torch.from_numpy(np.std(rewards_list, axis=0)).to(rewards.device).to(rewards.dtype)
                    advantages = (rewards - mean) / (std + 1e-4)

                    pro_advantages = []
                    for x, mu, log_sig in pro_result:
                        p = torch.exp(- F.mse_loss(mu, x, reduction='none') / (2 * (torch.exp(log_sig) ** 2)))
                        p = p / torch.exp(log_sig)
                        pro_advantages.append(p)
                    pro_advantages = torch.stack(pro_advantages, dim=1)
                    advantages = advantages.view(advantages.size(0), 1, 1, 1)
                    pro_advantages = pro_advantages * advantages
                    trg_idx = trg_idx.unsqueeze(-1)
                    trg_idx = trg_idx.unsqueeze(1)
                    trg_idx = trg_idx.repeat(1, pro_advantages.size(1), 1, pro_advantages.size(-1))
                    pro_advantages = pro_advantages[trg_idx]
                    pro_advantages = pro_advantages.mean()

                    loss_kl = reward.get_kl(pro_result, ref_pro_result)
                    loss_kl = loss_kl.mean()
                    loss = - pro_advantages + loss_kl
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
