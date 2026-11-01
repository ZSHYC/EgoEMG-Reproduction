"""RN18 fusion branch handoff entry points; full acceptance requires real150 fit."""
from .rn18_fusion_vision_producer import (
    LEGACY_FILES,LEGACY_SOURCES,READER_SOURCES,LegacyRn18Producer,Rn18FusionVisionLast,
    _validate_legacy_domains,_legacy_history,
    read_legacy_rn18_producer,admit_rn18_fusion_vision_last,read_rn18_fusion_vision_last,
)
from .rn18_fusion_trained_handoff import (
    Rn18FusionEmgBest,TrainedRn18FusionInitializers,read_rn18_fusion_emg_best,
    admit_rn18_fusion_initializers,read_rn18_fusion_initializers,
)
