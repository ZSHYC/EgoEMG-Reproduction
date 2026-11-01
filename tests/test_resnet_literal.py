"""Source-composed ResNet metadata; no data/model/GPU construction."""
import hashlib
import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("hydra")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf


def component():
    return importlib.import_module("egoemg_repro.resnet_literal")


def upstream():
    root = Path(os.environ.get("EGOEMG_TEST_UPSTREAM", Path(__file__).resolve().parents[1]/"upstream"))
    if not (root/"config/base.yaml").is_file():
        pytest.skip("pinned original upstream checkout required")
    return root


@pytest.mark.parametrize("variant,batch", [("resnet18",900),("resnet50",400),("resnet152",180)])
def test_exact_author_or_declared_derivative(variant,batch):
    lib=component();root=upstream();cfg=lib.compose_resnet(root,variant);p=lib.resnet_protocol(root,cfg)
    assert p["variant"]==variant and p["upstream_experiment"]=="fusion/vision_resnet18"
    assert p["module"]["_target_"]=="egoemg.models.modules.resnet_vision.ResNetVisionPose"
    assert p["module"]["backbone_type"]==variant and p["module"]["out_channels"]==22
    assert p["module"]["pretrained"] is True and p["module"]["freeze_backbone"] is False
    assert p["module"]["head_hidden"]==512 and p["module"]["head_dropout"]==.1
    assert p["batch_size_by_role"]==dict(train=batch,val=batch,test=batch)
    assert p["workers_by_role"]==dict(train=8,val=8,test=8)
    assert p["unused_declared_val_batch_size"]==256
    assert p["repeat_by_role"]==dict(train=2,val=2,test=2)
    assert p["window"]==dict(train_length=7790,eval_length=7790,train_stride=780,eval_stride=7790)
    assert p["vision_patch_size"]==256
    assert p["raw_loss_weights"]["landmark/fingertip"]==.01
    assert p["raw_loss_weights"]["fingertip_distance"]==.01
    assert p["effective_loss_weights"]["landmark/fingertip"]==.02
    assert p["optimizer"]["lr"]==1e-4 and p["optimizer"]["weight_decay"]==.01
    assert p["lr_scheduler"]["scheduler"]["T_max"]==150
    assert p["lr_scheduler"]["scheduler"]["eta_min"]==1e-5
    assert p["trainer"]["max_epochs"]==150 and p["trainer"]["check_val_every_n_epoch"]==2
    assert p["trainer"]["precision"]=="bf16-mixed" and p["trainer"]["gradient_clip_val"]==1
    es=next(x for x in p["callbacks"]if x["_target_"].endswith("EarlyStopping"))
    assert es["monitor"]=="val_mae" and es["mode"]=="min" and es["patience"]==30
    assert es["check_on_train_epoch_end"] is False
    assert p["seed"]==42 and p["trainer"]["devices"]==list(range(6))
    assert p["recipe_origin"]==("pinned_author_experiment"if variant=="resnet18"else"explicit_rn18_recipe_and_paper_batch_derivation")
    assert all(p[k] is False for k in("scientific_eligible","production_ready","full_paper_complete"))
    assert p["sources_sha256"]["config/experiment/fusion/vision_resnet18.yaml"]==hashlib.sha256((root/"config/experiment/fusion/vision_resnet18.yaml").read_bytes()).hexdigest()


@pytest.mark.parametrize("variant",["resnet34","resnet101","vit_small","wilor",None,True])
def test_unknown_or_wrong_family_rejected(variant):
    with pytest.raises(ValueError):component().compose_resnet(upstream(),variant)


@pytest.mark.parametrize("path,value",[
    ("module._target_","egoemg.models.modules.vit_vision.VisionViTPose"),
    ("module.backbone_type","resnet50"),("module.out_channels",20),
    ("module.pretrained",False),("module.head_hidden",256),("batch_size",256),
    ("num_workers",0),("trainer.precision","32-true"),
    ("trainer.check_val_every_n_epoch",1),("trainer.max_epochs",149),
    ("loss_weights.fingertip_distance",0.),("loss_weights.landmark/fingertip",0.),
    ("callbacks.2.patience",31),("callbacks.0.monitor","val_loss"),
    ("datamodule.window_length",12000),("vision_patch_size",224),
])
def test_bound_original_configuration_drift_rejected(path,value):
    lib=component();root=upstream();cfg=lib.compose_resnet(root)
    OmegaConf.update(cfg,path,value,force_add=True)
    with pytest.raises(ValueError):lib.resnet_protocol(root,cfg)


def test_binding_cannot_relabel_backbone():
    lib=component();root=upstream();cfg=lib.compose_resnet(root)
    cfg[lib.CONTEXT_KEY].variant="resnet50"
    with pytest.raises(ValueError):lib._validated(root,cfg)


def test_unbound_configuration_rejected():
    lib=component();root=upstream();cfg=lib.compose_resnet(root)
    raw=OmegaConf.to_container(cfg,resolve=True);raw.pop(lib.CONTEXT_KEY)
    with pytest.raises(ValueError):lib._validated(root,OmegaConf.create(raw))


def test_metadata_does_not_instantiate_objects(monkeypatch):
    lib=component();root=upstream()
    def forbidden(*args,**kwargs):raise AssertionError("metadata must not instantiate models, data or callbacks")
    monkeypatch.setattr("hydra.utils.instantiate",forbidden)
    p=lib.resnet_protocol(root,lib.compose_resnet(root,"resnet152"))
    assert p["variant"]=="resnet152" and p["scientific_eligible"] is False
