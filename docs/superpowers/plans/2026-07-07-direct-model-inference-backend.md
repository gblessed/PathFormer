# Direct Model Inference Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal backend that loads the direct Pathformer checkpoint for a scenario and generates multipaths for a batch of 6D prompts.

**Architecture:** Add one focused backend module that reconstructs the trained `PathDecoder`, resolves the checkpoint path, and wraps the existing batched autoregressive generation helper. Add one small unit test file that patches model loading and generation so the contract stays stable without requiring a real checkpoint.

**Tech Stack:** Python, PyTorch, unittest, unittest.mock

## Global Constraints

- Keep the implementation as simple and concise as possible.
- Match the direct training model configuration from `multiscenario_direct_training.py`.
- Accept full 6D prompts instead of building prompts from a DeepMIMO scenario.

---

### Task 1: Test The Backend Contract

**Files:**
- Create: `/home/blessedg/Pathformer/test_direct_model_inference_backend.py`
- Test: `/home/blessedg/Pathformer/test_direct_model_inference_backend.py`

**Interfaces:**
- Consumes: `DirectModelInferenceBackend.generate(prompts, max_generate_steps=25)`
- Produces: regression coverage for prompt validation and output wrapping

- [ ] Write a failing test for invalid prompt width and successful batched wrapping.
- [ ] Run `python -m unittest /home/blessedg/Pathformer/test_direct_model_inference_backend.py`.
- [ ] Verify the test fails because the backend does not exist yet.

### Task 2: Implement The Backend

**Files:**
- Modify: `/home/blessedg/Pathformer/direct_model_inference_backend.py`
- Test: `/home/blessedg/Pathformer/test_direct_model_inference_backend.py`

**Interfaces:**
- Consumes: `PathDecoder`, `generate_paths_no_env_batch`
- Produces: `DirectModelInferenceBackend`

- [ ] Implement checkpoint resolution, model construction, and lazy loading.
- [ ] Implement `generate(...)` with prompt validation and batched result wrapping.
- [ ] Re-run `python -m unittest /home/blessedg/Pathformer/test_direct_model_inference_backend.py`.
- [ ] Verify the test passes.
