# TMAE/PMTA OSF Preregistration

This document is the OSF-formatted preregistration companion for:
**Trust-Mediated Attractor Exploitation (TMAE) & Proxy-Mediated Trust Attack (PMTA), Experiment Design v1.0.1**.

## Registration Metadata
- Project: TMAE/PMTA
- Researcher: Vladimir Vasilenko (b102e)
- Date target: May-June 2026
- Related preprint: https://arxiv.org/abs/2604.12016

## Research Questions
- Can adversarial influence be transmitted through trust while attractor proximity remains stable?
- Is directional drift consistency (`M2`) a reliable detection signal for this class of attacks?
- Is proxy-mediated attack performance equivalent to human-mediated attack under practical effect bounds?

## Hypotheses
- H1: TMAE shows elevated `M2` after Phase 2 while `M1` remains stable.
- H2: `M2` onset appears earlier at deep layers than middle/early layers.
- H3: PMTA (`D`) is practically equivalent to human TMAE (`C`) under TOST.
- H4: Combined attack (`E`) yields layered M3 shallow + M2 deep signature.
- H5: No-cognitive-core agents show altered susceptibility profile.
- H6: Honest persuasion control (`H`) is a falsification condition for H1.

## Conditions
A, A-Long, B, C, D, E, F, G (baseline/TMAE/Honest), H, M7-control.

## Primary Outcome
- Session mean `M2` over final 10 turns at `L_deep`, comparing `C` vs `A-Long`.

## Statistical Plan
- Primary: Welch t-test / Wilcoxon / ANCOVA depending on assumptions.
- Secondary: Holm correction, log-rank survival for H2, TOST for H3, PERMANOVA for H4.

## Pilot Gates
- M6 attractor gate thresholds locked from pilot.
- Condition H micro-pilot gate before main collection.
- M7 sensitivity gate before main collection.

## Open Science
- Pre-registration published before data collection.
- Code and analysis in companion repository: https://github.com/b102e/tmae-pmta-experiment
