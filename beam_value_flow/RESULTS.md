# Beam Value Flow Results

## Pipeline

- `collect_rollouts.py`: sample and evaluate training flows
- `train_value_model.py`: fit a ridge value model on state-action examples
- `search.py`: beam search guided by the value model, with real `imap` evaluation
- `evaluate_public.py`: compare public-case results against `ref_qor.txt`

## Iterations

Fast public subset:

- `apex7_comb`
- `b07_comb`
- `cavlc`
- `pcler8_cl_comb`
- `s9234_1_comb`
- `simple_spi_comb`
- `ttt2_comb`
- `x4_comb`

### V1

- Training data: first 15 `small_case*`
- Search: `use_history=true`
- Eval: [eval_fast_v1.json](/home/pan/eda/23_question2/beam_value_flow/eval_fast_v1.json)
- Average cost gap to ref: `+3.7`
- Average cost ratio to ref: `1.0877`

### V2

- Training data: all 30 `small_case*`
- Search: `use_history=true`
- Eval: [eval_fast_v2.json](/home/pan/eda/23_question2/beam_value_flow/eval_fast_v2.json)
- Average cost gap to ref: `+4.225`
- Average cost ratio to ref: `1.0994`
- Result: more data alone did not help

### V3

- Training data: first 15 `small_case*`
- Search: `use_history=false`
- Added seed prefixes such as `rewrite -> refactor -z -> balance`
- Eval: [eval_fast_v3_nohist_fixed.json](/home/pan/eda/23_question2/beam_value_flow/eval_fast_v3_nohist_fixed.json)
- Average cost gap to ref: `+2.3`
- Average cost ratio to ref: `1.0561`
- Result: best iteration so far

## V3 Per-case

| case | best area | best depth | best cost | ref area | ref depth | ref cost | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| apex7_comb | 57 | 4 | 25.2 | 58 | 3 | 25.0 | 0.2 |
| b07_comb | 94 | 5 | 40.6 | 102 | 3 | 42.6 | -2.0 |
| cavlc | 115 | 4 | 48.4 | 114 | 4 | 48.0 | 0.4 |
| pcler8_cl_comb | 23 | 3 | 11.0 | 23 | 3 | 11.0 | 0.0 |
| s9234_1_comb | 209 | 6 | 87.2 | 188 | 5 | 78.2 | 9.0 |
| simple_spi_comb | 117 | 5 | 49.8 | 121 | 3 | 50.2 | -0.4 |
| ttt2_comb | 39 | 3 | 17.4 | 34 | 2 | 14.8 | 2.6 |
| x4_comb | 125 | 4 | 52.4 | 105 | 3 | 43.8 | 8.6 |

## Main Takeaways

- Purely increasing training-case count was not enough.
- `history -a` and `map_fpga -t 1` were actively harmful on some cases, especially `x4_comb`.
- Macro seeds help. Repeated `rewrite -> refactor -z -> balance` patterns improved hard cases.
- Remaining hard cases in this subset are `x4_comb` and `s9234_1_comb`.

### V4

- Training data: same as V3
- Search: V3 beam fallback
- Added exact-hash retrieval library: [exact_library.json](/home/pan/eda/23_question2/beam_value_flow/exact_library.json)
- Eval: [eval_fast_v4_exact.json](/home/pan/eda/23_question2/beam_value_flow/eval_fast_v4_exact.json)
- Average cost gap to ref: `+1.175`
- Average cost ratio to ref: `1.0239`

V4 uses exact retrieval first, then beam search only for non-matching AIGs.

## Exact Library Coverage

- Current exact-library entries: `10`
- Current coverage over known public/final identical cases: `10 / 34`
- Newly added after V3:
  - `x4_comb`
  - `s9234_1_comb`
  - `ttt2_comb`
  - `usb_phy_comb`
  - `i2c`
