from importlib.resources import files

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset
from rl import trainer_rl


# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'

tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
dataset_name = "RL_ZH"

# -------------------------- Training Settings -------------------------- #

exp_name = "F5TTS_Base_rl"  # F5TTS_Base | E2TTS_Base

learning_rate = 1e-5

batch_size_per_gpu = 1600
batch_size_type = "frame"  # "frame" or "sample"
max_samples = 1  # max sequences per batch if use frame-wise batch_size.
grad_accumulation_steps = 8  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 5  # use linear decay, thus epochs control the slope
num_warmup_updates = 100  # warmup steps
save_per_updates = 100  # save checkpoint per steps
last_per_steps = 100  # save last checkpoint per steps

# model params
wandb_resume_id = None
model_cls = DiT
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

# ----------------------------------------------------------------------- #


def main():
    if tokenizer != "custom":
        tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    e2tts = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = trainer_rl.GRPOTrainer(
        e2tts,
        epochs,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../ckpts/{exp_name}")),
        batch_size=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=last_per_steps,
    )

    train_dataset = load_dataset(dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(
        train_dataset,
        resumable_with_seed=666  # seed for shuffling dataset
    )


if __name__ == '__main__':
    main()
