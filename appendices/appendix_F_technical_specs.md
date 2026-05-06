# Appendix F — Technical Specifications

Related plan section: Appendix F (TMAE/PMTA Experiment Design v1.0.1).

## F.1 Activation log format
- Per session, per layer: array of shape `[n_turns, d_model]` (Gemma 4: 5120; Llama 3.1: 4096).
- Final pooling strategy determined post-pilot; default output-token-only if mean-pool differs by >0.05.
- Three files per session:
  - `activations_L_early.npy`
  - `activations_L_mid.npy`
  - `activations_L_deep.npy`
- Metadata JSON per session includes:
  - `condition`, `model`, `session_id`, `turn_count`, `M6_passed`, `phase_transitions`, `M7_pre`, `M7_post`, `pooling_strategy`.

## F.2 Random seed protocol
- All stochastic operations use `random_seed = 20260506`, committed before first pilot session.
- Applies to permutation tests, bootstrap, and model inference when deterministic mode is available.

## F.3 M2 detector verifier protocol
- Detector uses frozen baseline checkpoint and thresholds derived exclusively from Condition A-Long logs.
- No recalibration after main collection starts.
- Verifier code and checkpoint hash are published in the experiment repository.
