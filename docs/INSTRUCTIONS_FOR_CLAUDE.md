# Instructions for Claude (AWS Training Instance)

Paste the relevant block into Claude when you SSM into the p4d training instance. Each block is a complete, self-contained instruction.

---

## Session 1: Evaluate existing female 100K checkpoint

Use this when you want to measure the current female voice quality with the new 55-prompt eval pack before deciding whether to retrain.

```
Pull latest changes from the codex/voice-pipeline branch and follow docs/VOICE_PIPELINE.md.

Goal: Evaluate the existing female 100K checkpoint using the new pipeline.

- The checkpoint is in S3 at s3://pm-tts-model/checkpoints/female/GPT_XTTS_v2.0_LJSpeech_FT-April-10-2026_10+13AM-3c5f7c31/
- Download the best_model.pth, config.json, vocab.json, dvae.pth, and mel_stats.pth if not already local
- Run scripts/eval/generate_samples.py with the configs/eval/english_core.json prompt pack (55 prompts)
- Run scripts/eval/automated_metrics.py on the resulting manifest for WER scores
- Sync the sample outputs and metrics back to S3 at s3://pm-tts-model/evals/female_100k/

Don't retrain anything. Only evaluate the existing checkpoint with the new pipeline.
```

---

## Session 2: Train male voice (Hi-Fi TTS 6097)

Use this after you've evaluated the female voice and decided to proceed to male training.

```
Follow docs/VOICE_PIPELINE.md to prepare the Hi-Fi TTS speaker 6097 dataset and train the male voice.

Preparation:
- Download Hi-Fi TTS speaker 6097 from OpenSLR 109 to /home/ec2-user/data/hi_fi_tts
- Run scripts/data/prepare_hifi_tts.py to generate filtered train/eval manifests at /home/ec2-user/data/prepared/hifi_tts_6097
- Pick 3-5 clean reference clips (6-10 seconds each) and place them at /home/ec2-user/TTS/reference_audio/male/

Config changes before training:
- In configs/voices/male_hifi_tts_6097.json, set "sync_to_s3_after_train": true
- Verify "reference_audio" points to the 3-5 clips you selected
- Use the STOCK XTTS v2 pretrained weights as the baseline (not the female 100K model — we want generic starting weights, not female-specialized)

Training:
- Run torchrun --nproc_per_node=8 scripts/training/train_male.py
- Target 100K steps (matches the female voice)
- Save checkpoints every 5K steps

Post-training:
- Run the eval pipeline on the trained male checkpoint: generate_samples.py + automated_metrics.py
- Sync samples and metrics to s3://pm-tts-model/evals/male_100k/
- Verify the S3 sync from voice_training.py completed successfully
- Run sudo shutdown -h now to stop the instance
```

---

## Session 3: Extend female training (if the eval showed it needs more work)

Use this if after Session 1 you listened to the samples and decided the female voice needs more steps.

```
Follow docs/VOICE_PIPELINE.md to resume female voice training from the existing 100K checkpoint.

- Set XTTS_BASELINE_CHECKPOINT to s3://pm-tts-model/checkpoints/female/GPT_XTTS_v2.0_LJSpeech_FT-April-10-2026_10+13AM-3c5f7c31/best_model.pth (download locally first)
- In configs/voices/female_ljspeech.json or female_ljspeech_prepared.json, set "sync_to_s3_after_train": true
- Bump the target epochs so training continues past 100K steps (aim for 150K total)
- Run torchrun --nproc_per_node=8 scripts/training/train_female.py
- The training script should auto-resume from the latest checkpoint in the output directory, or use the baseline if none exists

Post-training:
- Run the eval pipeline on the new checkpoint
- Sync to s3://pm-tts-model/evals/female_150k/
- Run sudo shutdown -h now
```

---

## Notes

- All training runs must use `torchrun --nproc_per_node=8` to use all 8 GPUs on the p4d.24xlarge.
- The instance auto-shutdown is **not** built into `voice_training.py` anymore — you must `sudo shutdown -h now` manually or bake it into the instruction above.
- The S3 sync is controlled by `"sync_to_s3_after_train": true` in the voice config file. Without this, checkpoints stay local on the instance.
- If you want to stop manually mid-training, the latest checkpoint will already be on disk and syncing later is fine via `aws s3 sync /home/ec2-user/training_output/ s3://pm-tts-model/checkpoints/`.
- Cost reminder: p4d.24xlarge is ~$32/hr on-demand. A full train + eval session is ~5-7 hours (~$160-230). Don't leave it running idle.
