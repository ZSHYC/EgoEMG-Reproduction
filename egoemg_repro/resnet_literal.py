"""Pinned RN18 recipe and explicitly identified RN50/152 derivatives.

This module only composes and validates metadata. It constructs no model,
datamodule, optimizer or CUDA object and grants no scientific permit.
"""
from __future__ import annotations

from . import UPSTREAM_COMMIT
from .classic_emg import _approved_upstream, _file_hash, _hash


VARIANTS = ("resnet18", "resnet50", "resnet152")
CONTEXT_KEY = "_literal_resnet"
EXPERIMENT = "fusion/vision_resnet18"
PAPER_BATCHES = {"resnet50": 400, "resnet152": 180}
SEED_POLICY = "original setup_runtime before original factory; metadata constructs no runtime"
SOURCE_FILES = (
    "config/base.yaml", "config/lineage/_base.yaml", "config/lineage/fusion.yaml",
    "config/module/resnet_vision.yaml", "config/experiment/fusion/vision_resnet18.yaml",
    "config/dataset/egoemg_unified_angle_regression.yaml", "config/datamodule/default.yaml",
    "config/lr_scheduler/cosine.yaml", "config/transforms/noop.yaml",
    "egoemg/datamodule.py", "egoemg/lightning.py", "egoemg/train.py",
    "egoemg/models/modules/resnet_vision.py",
)


def _variant(value):
    if type(value) is not str or value not in VARIANTS:
        raise ValueError("unsupported original ResNet variant")
    return value


def _overrides(variant):
    if variant == "resnet18":
        return []
    return ["module.backbone_type="+variant, "batch_size="+str(PAPER_BATCHES[variant])]


def _binding(root, variant):
    return dict(format="original_resnet_config_binding_v1", upstream=str(root),
                variant=variant, seed_policy=SEED_POLICY, explicit_overrides=_overrides(variant))


def _compose(root, variant):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/"config"), version_base="1.1"):
        config = compose(config_name="base", overrides=["experiment="+EXPERIMENT, *_overrides(variant)])
    OmegaConf.resolve(config)
    return config


def compose_resnet(upstream, variant="resnet18"):
    """Compose author RN18, or RN18 plus a named backbone and paper batch."""
    from omegaconf import OmegaConf
    variant = _variant(variant)
    root = _approved_upstream(upstream)
    config = _compose(root, variant)
    OmegaConf.update(config, CONTEXT_KEY, _binding(root, variant), force_add=True)
    return config


def _validated(upstream, config):
    from omegaconf import OmegaConf
    root = _approved_upstream(upstream)
    try:
        binding = OmegaConf.to_container(config[CONTEXT_KEY], resolve=True)
        variant = _variant(binding["variant"])
        if _hash(binding) != _hash(_binding(root, variant)):
            raise ValueError("ResNet configuration source binding differs")
        actual = OmegaConf.to_container(config, resolve=True)
        del actual[CONTEXT_KEY]
        expected = OmegaConf.to_container(_compose(root, variant), resolve=True)
        if _hash(actual) != _hash(expected):
            raise ValueError("ResNet configuration differs from author recipe/explicit derivative")
        return root, variant, actual
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid original ResNet configuration binding") from exc


def resnet_protocol(upstream, config):
    """Describe original execution rules without constructing execution objects."""
    root, variant, value = _validated(upstream, config)
    effective = {}
    for name, weight in value["loss_weights"].items():
        key = {"fingertip_distance": "landmark/fingertip", "landmark_distance": "landmark/all"}.get(name, name)
        effective[key] = effective.get(key, 0.) + float(weight)
    dm = value["datamodule"]
    eval_workers = dm.get("eval_num_workers")
    if eval_workers is None:
        eval_workers = value["num_workers"]
    roles = ("train", "val", "test")
    result = dict(
        format="original_resnet_execution_contract_v1", variant=variant,
        upstream_revision=UPSTREAM_COMMIT, upstream_experiment=EXPERIMENT,
        recipe_origin="pinned_author_experiment" if variant=="resnet18" else "explicit_rn18_recipe_and_paper_batch_derivation",
        explicit_overrides=_overrides(variant), module=value["module"],
        raw_loss_weights=value["loss_weights"], effective_loss_weights=effective,
        optimizer=value["optimizer"], lr_scheduler=value["lr_scheduler"],
        trainer=value["trainer"], callbacks=value["callbacks"], seed=value["seed"], seed_policy=SEED_POLICY,
        batch_size_by_role={r: value["batch_size"] for r in roles},
        unused_declared_val_batch_size=value["val_batch_size"],
        batch_size_semantics="native make_data_module passes global batch_size; all roles use self.batch_size",
        workers_by_role=dict(train=value["num_workers"], val=eval_workers, test=eval_workers),
        loader=dict(prefetch_factor=dm["prefetch_factor"], persistent_workers=dm["persistent_workers"],
                    pin_memory=dm["pin_memory"], generator=None, drop_last=False),
        repeat_by_role={r: dm["dataset_repeat"] for r in roles},
        window=dict(train_length=dm["window_length"], eval_length=dm["val_test_window_length"],
                    train_stride=dm["stride"], eval_stride=dm["val_test_stride"]),
        vision_patch_size=value["vision_patch_size"], dataset_declarations=value["dataset"],
        declared_training_corpus="physically merged EgoEMG/ShowEE/Incre; no corpus admitted here",
        mask_semantics="original generated center label mask for22 angles; no legacy wrist cleanup",
        sources_sha256={n: _file_hash(root/n) for n in SOURCE_FILES},
        scientific_eligible=False, production_ready=False, full_paper_complete=False,
    )
    result["sha256"] = _hash(result)
    return result
