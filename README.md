# Themis

Themis is a planned text-only, general-purpose decision model built from
`Qwen/Qwen3.5-0.8B`. It takes a case, runtime-defined questions, and allowed
answers, and returns probabilities for each decision.

Project directory: `/home/lothnic/Desktop/Research/themis/`.

## Initial scope

- Support yes/no (`noul`), pick-one (`choice`), and ordered-rating (`score`) questions.
- Evaluate generalization to new workflows and question schemas.
- Train locally on an NVIDIA RTX PRO 2000 Blackwell with 16 GB VRAM.
- Start with BF16 LoRA and a trainable, schema-conditioned decision head.
- Establish a Clef-style baseline before comparing an AMOS-style option-token readout
  and bidirectional full attention.

## Candidate training sources

| Dataset | Intended role |
| --- | --- |
| [Tasksource Jev Typed Decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | Broad annotated task mixture; inspect source terms and select text records |
| [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | Explicit rules, eligibility, arithmetic, tables, and state tracking |
| [Synthetic Typed Decisions](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions) | Workflow judgments and soft target distributions |
| [LocalLLaMA Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Multi-question customer service, invoices, security, and agent observability |

Use [Decision Bench](https://huggingface.co/datasets/atlanai/decision-bench) as an
evaluation source and coverage reference. Keep its records out of training,
including duplicates inherited through other collections. Evaluate text-only
cases separately from image-dependent tasks.

## Proposed first mixture

Target 50,000 decision instances, where one instance means one question and its
target. A case may contain several instances. These are planning targets, not an
assembled dataset.

| Category | Decisions |
| --- | ---: |
| Engineering | 7,000 |
| AI agents and evaluations | 7,000 |
| Trust and safety | 4,000 |
| Customer support | 5,000 |
| Sales and commerce | 4,000 |
| Finance | 5,000 |
| Legal | 5,000 |
| Product management | 3,000 |
| Data and analytics | 5,000 |
| Documents and meetings | 4,000 |
| Design: text-based UI and accessibility rules | 1,000 |
| **Total** | **50,000** |

## Proposed initial training settings

| Setting | Starting value |
| --- | --- |
| Precision | BF16 |
| Context length | 2,048 tokens |
| Microbatch | 2 |
| Gradient accumulation | 8 |
| LoRA rank | 32 |
| Adapter learning rate | 1e-4 |
| Decision head learning rate | 3e-4 |
| Epochs | 1, then select further training by validation |
| Gradient checkpointing | Enabled |

These settings require a local memory and compatibility check before a full run.

## Implementation sequence

1. Inspect local GPU/software support and pin a compatible training environment.
2. Inspect source schemas, revisions, licenses, and representative records.
3. Implement normalization, provenance tracking, sampling, and contamination checks.
4. Split related records together; reserve unseen workflows and schemas for testing.
5. Run a small training and inference smoke test, then the first mixture experiment.
6. Fit probability calibration on a separate calibration split.
7. Evaluate accuracy, NLL/Brier score, calibration, option-order sensitivity,
   selective prediction, latency, and peak VRAM.

Preserve source soft labels. Keep labels and privileged metadata outside model
inputs. For any bidirectional experiment, keep gold answers out of the input
sequence and use answer placeholders instead.
