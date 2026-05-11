# RL Flow

PyTorch-based policy optimization flow for EDA23 problem 2.

This version no longer uses offline DQN or replay-buffer imitation. It treats `iMAP` as a short-horizon RL environment and optimizes the final objective directly:

- state: current AIG plus probe-mapped QoR features
- action: macro optimization commands plus terminal mapping actions
- target: minimize final `0.6 * depth + 0.4 * area`
- training: pairwise ranking on candidate trajectories plus elite imitation
- inference: policy-guided beam search instead of one-step greedy decoding

## Layout

- `actions.py`: macro action definitions
- `imap_env.py`: `imap -c` environment wrapper with branchable search states
- `model.py`: policy/value network
- `policy_search.py`: rollout sampling, checkpoint loading, policy-guided beam search
- `train_policy.py`: end-to-end training on a split
- `infer_seq.py`: emit a `.seq` for one `.aig`
- `evaluate_policy.py`: evaluate a checkpoint on a split
- `split_cases.py`: split cases into train/eval/test

## Dependencies

- Python 3.10+
- `numpy`
- `torch`
- optional: `tqdm`
- compiled `iMAP` binary at `iMAP/bin/imap`

## Recommended workflow

1. Split the public set:

```bash
/home/pan/miniconda3/envs/iMap/bin/python rl_flow/split_cases.py \
  --case-root /home/pan/eda/23_question2/iMAP/eda23/benchmark_public \
  --output /home/pan/eda/23_question2/rl_flow/public_split.json
```

2. Train the policy:

```bash
/home/pan/miniconda3/envs/iMap/bin/python rl_flow/train_policy.py \
  --split /home/pan/eda/23_question2/rl_flow/public_split.json \
  --split-name train \
  --eval-split-name eval \
  --case-root /home/pan/eda/23_question2/iMAP/eda23/benchmark_public \
  --imap-bin /home/pan/eda/23_question2/iMAP/bin/imap \
  --output /tmp/imap_policy.pt \
  --history-json /tmp/imap_policy_history.json \
  --epochs 30 \
  --episodes-per-case 4 \
  --max-steps 4 \
  --num-workers 4 \
  --eval-workers 4 \
  --beam-width 5 \
  --beam-branch-topk 4 \
  --device auto
```

3. Evaluate the checkpoint:

```bash
/home/pan/miniconda3/envs/iMap/bin/python rl_flow/evaluate_policy.py \
  --split /home/pan/eda/23_question2/rl_flow/public_split.json \
  --split-name eval \
  --case-root /home/pan/eda/23_question2/iMAP/eda23/benchmark_public \
  --checkpoint /tmp/imap_policy.pt \
  --imap-bin /home/pan/eda/23_question2/iMAP/bin/imap \
  --output-json /tmp/imap_eval.json \
  --num-workers 4 \
  --beam-width 5 \
  --beam-branch-topk 4 \
  --device auto
```

4. Generate a `.seq`:

```bash
/home/pan/miniconda3/envs/iMap/bin/python rl_flow/infer_seq.py \
  /home/pan/eda/23_question2/iMAP/eda23/benchmark_public/C432.iscas_comb/C432.iscas_comb.aig \
  /tmp/C432.seq \
  --checkpoint /tmp/imap_policy.pt \
  --imap-bin /home/pan/eda/23_question2/iMAP/bin/imap \
  --beam-width 5 \
  --beam-branch-topk 4 \
  --device auto
```

## Practical notes

- `episodes-per-case` controls exploration effort during training.
- `num-workers` controls parallel rollout collection across cases.
- `eval-workers` controls parallel evaluation during training; if omitted it follows `num-workers`.
- `epsilon` controls random exploration during sampling.
- `min-return-gap`, `max-pairs-per-bucket`, `ranking-margin` control the pairwise ranking dataset.
- `elite-topk-per-case` controls how many strong trajectories per case feed imitation learning.
- `beam-width` and `beam-branch-topk` matter at inference time; the final `.seq` quality depends on search, not just one forward pass.
- `ImapEnv` now caches repeated `case + sequence + map_command` evaluations automatically inside each process.
- The important training metrics are:
  - `train_avg_return`
  - train/eval average final cost
  - `eval_avg_ref_gap`
  - `eval_exact_ref_matches`
  - best cost found per case
