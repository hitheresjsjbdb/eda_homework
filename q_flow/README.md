# Q Flow

`q_flow` is a new offline training + fast inference framework for EDA23 question 2.

Core direction:

- heavy teacher search during dataset build
- cached offline training after teacher build
- fast greedy inference at eval/test time
- only terminal `map_fpga`, no repeated intermediate mapping in rollout states

This is intentionally different from `rl_flow`. The model is trained to choose actions directly, instead of depending on beam search at inference time.

## Main idea

`q_flow` separates the problem into two phases:

1. Teacher build
   - bounded search over AIG transforms
   - terminal-only FPGA mapping
   - produce per-state best action and Q targets
   - cache records to disk per case
2. Offline policy training
   - pure neural training on cached teacher data
   - no repeated environment collection every epoch
3. Fast inference
   - greedy action selection by the Q model
   - tiny local fallback only when model confidence is low

This gives the main speed win:

- first run pays teacher-build cost once
- later training runs reuse cache and become much faster
- eval/test no longer require deep beam search

## Files

- `actions.py`: transform actions and terminal map actions
- `env.py`: AIG feature extraction and terminal QoR evaluation
- `teacher.py`: bounded terminal-search teacher
- `model.py`: greedy Q network
- `inference.py`: shared fast inference with low-confidence fallback
- `train_q_policy.py`: offline training with case-level cache
- `infer_greedy.py`: generate a `.seq`
- `evaluate_greedy.py`: split evaluation using the same inference logic

## Recommended train command

First build cache and train:

```bash
/home/pan/miniconda3/envs/iMap/bin/python q_flow/train_q_policy.py \
  --split ./rl_flow/public_split.json \
  --split-name train \
  --case-root ./iMAP/eda23/benchmark_public \
  --imap-bin ./iMAP/bin/imap \
  --output ./tmp/q_flow_main.pt \
  --history-json ./tmp/q_flow_main.json \
  --cache-dir ./tmp/q_flow_cache_main \
  --epochs 20 \
  --max-steps 4 \
  --branch-topk 4 \
  --terminal-topk 2 \
  --num-workers 16 \
  --batch-size 256 \
  --timeout-sec 60 \
  --device auto
```

Then rerun the same command for later experiments. It will reuse cache automatically and skip most of the teacher cost.

## Inference

```bash
/home/pan/miniconda3/envs/iMap/bin/python q_flow/infer_greedy.py \
  ./iMAP/eda23/benchmark_public/des_area_comb/des_area_comb.aig \
  ./tmp/des_area_comb_q.seq \
  --checkpoint ./tmp/q_flow_main.pt \
  --imap-bin ./iMAP/bin/imap \
  --max-steps 4 \
  --confidence-margin 0.08 \
  --fallback-topk 3 \
  --fallback-depth 2 \
  --device auto
```

## Evaluation

```bash
/home/pan/miniconda3/envs/iMap/bin/python q_flow/evaluate_greedy.py \
  --split ./rl_flow/public_split.json \
  --split-name eval \
  --case-root ./iMAP/eda23/benchmark_public \
  --checkpoint ./tmp/q_flow_main.pt \
  --imap-bin ./iMAP/bin/imap \
  --max-steps 4 \
  --confidence-margin 0.08 \
  --fallback-topk 3 \
  --fallback-depth 2 \
  --timeout-sec 60 \
  --output-json ./tmp/q_flow_eval.json \
  --device auto
```

## What improves speed most

- cache teacher records per case
- parallel teacher build with `--num-workers`
- terminal-only mapping during search
- eval with greedy policy instead of beam search

## What improves quality most

- richer AIG features from `print_stats`
- stronger terminal-search teacher instead of one-step probe labels
- direct action learning with a small fallback instead of large online search
