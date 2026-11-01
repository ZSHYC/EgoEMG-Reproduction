"""RN18 transfer receipt checks without changing the frozen cotrain provider.

The complete tensor-inventory checks preserve fusion_literal_runtime's existing
rules. That module is part of the actual completed producer's source identity;
keep its bytes unchanged. RN18 separately verifies its full trained lineage and
passes the required provenance flag here, without rewriting saved metadata.
"""
from .fusion_literal_runtime import _PREFIXES


def saved_rn18_transfer(identity,*,expected_training_provenance=False):
    """Require the complete transfer receipt, beyond its content checksum."""
    if type(expected_training_provenance)is not bool:
        raise ValueError('explicit RN18 training-provenance policy required')
    import torch
    def digest(value):
        return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    try:
        init=identity['initialization'];schema=identity['model_schema']
        if (type(identity['parameters']) is not int or identity['parameters']<=0 or
            init['initializer_training_provenance_verified'] is not expected_training_provenance):raise ValueError('invalid fusion initialization scope')
        for role in ('vision_file','emg_file'):
            entry=init[role]
            if (type(entry['bytes']) is not int or entry['bytes']<=0 or
                not all(digest(entry[k]) for k in ('sha256','state_sha256'))):raise ValueError('invalid branch file receipt')
        vision=init['vision_transfer']
        for label,prefix in (('backbone','model.vision_backbone.'),('head','model.head_vision.')):
            count=sum(k.startswith(prefix) for k in schema)
            if count<=0 or type(vision[label+'_tensors']) is not int or vision[label+'_tensors']!=count:
                raise ValueError('vision transfer receipt differs from complete schema')
            if not all(digest(vision[k]) for k in ('source_'+label+'_state_sha256','loaded_'+label+'_state_sha256')):
                raise ValueError('invalid vision tensor receipt')
        emg=init['emg_transfer'];matched=emg['matched'];skipped=emg['skipped']
        if (emg['strict'] is not False or emg['weight_expansion_performed'] is not False or
            not isinstance(matched,list) or not isinstance(skipped,list) or
            not all(any(x['target_key'].startswith(prefix) for x in matched) for prefix in ('featurizer.','decoder.'))):
            raise ValueError('incomplete native EMG transfer receipt')
        inventory=init['emg_source_schema'];normalized={}
        if not isinstance(inventory,dict) or not inventory:raise ValueError('complete EMG source inventory required')
        for source,descriptor in inventory.items():
            if type(source) is not str or not source or not isinstance(descriptor,dict):
                raise ValueError('invalid EMG source descriptor')
            shape=descriptor['shape'];dtype=descriptor['dtype']
            if (type(shape) is not list or any(type(n) is not int or n<0 for n in shape) or
                type(dtype) is not str or not dtype.startswith('torch.') or
                not isinstance(getattr(torch,dtype[6:],None),torch.dtype)):
                raise ValueError('invalid EMG source shape/dtype')
            target=source[6:] if source.startswith('model.') else source
            if not target or target in normalized:raise ValueError('ambiguous normalized EMG source inventory')
            normalized[target]=source
        seen=set()
        for loaded,items in ((True,matched),(False,skipped)):
            for item in items:
                source=item['source_key'];target=item['target_key']
                if source not in inventory or source in seen or normalized.get(target)!=source:
                    raise ValueError('duplicate or inconsistent EMG source transfer receipt')
                seen.add(source);descriptor=inventory[source];destination=schema.get('model.'+target)
                if item['source_shape']!=descriptor['shape'] or item['source_dtype']!=descriptor['dtype']:
                    raise ValueError('EMG transfer source differs from inventory')
                if not target.startswith(_PREFIXES):reason='not_backbone_prefix'
                elif destination is None:reason='missing_target'
                elif descriptor['shape']!=destination['shape']:reason='shape_mismatch'
                else:reason=None
                if loaded:
                    if (reason is not None or item['target_dtype']!=destination['dtype'] or
                        not digest(item['loaded_sha256'])):
                        raise ValueError('matched EMG receipt differs from native loading rule')
                elif (reason is None or item['reason']!=reason or
                    item['target_shape']!=(destination['shape'] if destination is not None else None)):
                    raise ValueError('skipped EMG receipt differs from native loading rule')
        if seen!=set(inventory):raise ValueError('EMG transfer receipt does not cover every source tensor')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('incomplete fusion initialization receipt') from exc
