"""Audit pinned source and render descriptive paper Figures 5 and 6 (no Torch)."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

REVISION = 'f7cbd048b7092139852270491d3859cd472e0311'
FORMAT = 'egoemg_pinned_vision_fusion_architecture_v1'
MODULES = 'egoemg/models/modules/'
SOURCE_FILES = [
    MODULES + name for name in (
        '_vision_constants.py', 'resnet_vision.py', 'vit_vision.py',
        'wilor_vit.py', 'mid_fusion.py', '_pooling.py',
    )
] + [
    'egoemg/models/heads/mlp.py',
    'egoemg/models/featurizers/tds.py',
    'config/module/head/mlp.yaml',
    'config/module/featurizer/tds_slim_16ch.yaml',
    'config/module/mid_fusion.yaml',
    'config/module/decoder/transformer/preset/small.yaml',
    'config/experiment/fusion/fusion_vits_s_center_16ch_wl7790.yaml',
]
EVIDENCE_FILES = [
    'results/native-fusion-construction-initialization-20261028.json',
    'results/native-fusion-construction-cpu-20261028.json',
    'results/native-fusion-full-first-epoch-20261028.json',
    'results/dataset-statistics/emgformer-architecture.json',
    'results/vision-literal-gpu-engineering.json',
    'results/vision-literal-base-gpu-engineering.json',
    'results/vision-literal-large-checkpoint-gpu-engineering.json',
    'results/wilor-native-model-cpu-20261028.json',
    'results/wilor-native-gpu-qualification-review-20261029.json',
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def member(tree, class_name, method):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)


def assignment(node, target):
    return next(n.value for n in ast.walk(node) if isinstance(n, ast.Assign)
                and any(ast.unparse(t) == target for t in n.targets))


def head_info(tree, class_name):
    init = member(tree, class_name, '__init__')
    defaults = dict(zip([n.arg for n in init.args.args][-len(init.args.defaults):],
                        init.args.defaults))
    head = assignment(init, 'self.head')
    assert ast.unparse(head.func) == 'nn.Sequential'
    layers = [ast.unparse(n.func) for n in head.args]
    assert layers[0] == layers[3] == 'nn.Linear' and layers[2] == 'nn.Dropout'
    assert [ast.unparse(n) for n in head.args[3].args] == ['head_hidden', 'out_channels']
    return dict(layers=layers, hidden=ast.literal_eval(defaults['head_hidden']),
                dropout=ast.literal_eval(defaults['head_dropout']),
                first_linear_arguments=[ast.unparse(n) for n in head.args[0].args],
                activation=layers[1].removeprefix('nn.'))


def audit(upstream, repo):
    upstream = upstream.resolve()
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain',
                                        '--untracked-files=all'], text=True).strip()
    source = {}
    for name in SOURCE_FILES:
        data = (upstream / name).read_bytes()
        original = subprocess.check_output(['git', '-C', str(upstream), 'show', REVISION + ':' + name])
        assert data == original
        source[name] = sha(upstream / name)
    trees = {n: ast.parse((upstream / (MODULES + n + '.py')).read_text())
             for n in ('_vision_constants', 'resnet_vision', 'vit_vision', 'wilor_vit', 'mid_fusion', '_pooling')}
    constants = trees['_vision_constants']
    resnet_dims = ast.literal_eval(assignment(constants, 'RESNET_DIMS'))
    dino_dims = ast.literal_eval(assignment(constants, 'DINOV2_VARIANTS'))
    heads = {name: head_info(trees[name], cls) for name, cls in (
        ('resnet_vision', 'ResNetVisionPose'), ('vit_vision', 'VisionViTPose'),
        ('wilor_vit', 'WiLoRViTPose'))}
    assert [heads[n]['activation'] for n in heads] == ['ReLU', 'ReLU', 'GELU']
    assert all(v['hidden'] == 512 and v['dropout'] == .1 for v in heads.values())
    assert heads['wilor_vit']['first_linear_arguments'] == ['1280', 'head_hidden']
    dino_forward = member(trees['vit_vision'], 'VisionViTPose', 'forward')
    assert ast.unparse(assignment(dino_forward, 'feat')) == 'self.backbone(img)'
    crop = member(trees['wilor_vit'], 'WiLoRViTPose', '_extract_feature')
    assert ast.unparse(assignment(crop, 'images')) == 'images[:, :, 32:-32, :]'
    init_residual = next(n for n in trees['mid_fusion'].body
                         if isinstance(n, ast.FunctionDef) and n.name == '_init_residual_branch')
    init_calls = {ast.unparse(n.func): n for n in ast.walk(init_residual) if isinstance(n, ast.Call)}
    assert {k.arg: ast.literal_eval(k.value) for k in init_calls['nn.init.normal_'].keywords} == dict(mean=0., std=1e-5)
    assert ast.unparse(init_calls['nn.init.zeros_'].args[0]) == 'last_linear.bias'
    evidence = {name: json.loads((repo / name).read_text()) for name in EVIDENCE_FILES}
    construction = evidence[EVIDENCE_FILES[1]]
    protocol = construction['protocol']
    cfg = protocol['module']
    assert protocol['upstream_revision'] == REVISION
    assert cfg['fusion_mode'] == 'center_supervised' and cfg['vision_backbone_type'] == 'vit_small'
    assert cfg['head']['activation'] == 'relu' and cfg['head']['hidden_sizes'] == [512]
    assert cfg['head']['dropout'] == .1 and cfg['fusion_proj_dim'] == 256
    assert protocol['layout']['input_channels'] == 16
    assert protocol['window']['train_length'] == protocol['window']['eval_length'] == 7790
    for name, recorded_sha in construction['source_files_sha256'].items():
        if name.startswith('upstream/') and name[9:] in source:
            assert source[name[9:]] == recorded_sha
    init = evidence[EVIDENCE_FILES[0]]
    expected_shapes = {
        'vision_proj.weight': [256, 384],
        'fusion_proj.0.weight': [512, 512, 1], 'fusion_proj.3.weight': [256, 512, 1],
        'head.net.0.weight': [512, 256], 'head.net.3.weight': [22, 512],
        'head_vision.0.weight': [512, 384], 'head_vision.3.weight': [22, 512],
        'temporal_attn.0.weight': [64, 256], 'temporal_attn.2.weight': [1, 64],
    }
    assert all(init['model_schema']['model.' + key]['shape'] == shape
               for key, shape in expected_shapes.items())
    first_epoch = evidence[EVIDENCE_FILES[2]]
    handoff = first_epoch['actual_completed_trained_branch_identity']
    assert first_epoch['actual_handoff_and_independent_new_process_reopen_equal'] is True
    assert handoff['producer_training_verified'] is True
    assert first_epoch['actual_first_complete_epoch']['train_samples'] == 395276
    assert handoff['emg']['completed_epochs'] == 150 and handoff['emg']['selected_epoch'] == 133
    assert handoff['vision']['completed_epochs'] == 124
    # Derive only temporal lengths from the pinned TDS convolution configuration.
    # The prior measured 8-channel diagnostic independently checks this geometry.
    lengths = [7790]
    featurizer = cfg['featurizer']
    for conv in featurizer['conv_blocks']:
        lengths.append((lengths[-1] - conv['kernel_size']) // conv['stride'] + 1)
    for stage in featurizer['tds_stages']:
        lengths.append((lengths[-1] - stage['in_conv_kernel_width']) // stage['in_conv_stride'] + 1)
        for _ in range(stage['num_blocks']):
            lengths.append(lengths[-1] - stage['kernel_width'] + 1)
    assert lengths == [7790, 1556, 776, 154, 150, 148, 146]
    measured = evidence[EVIDENCE_FILES[3]]['models'][0]['forwards'][1]
    assert measured['input_shape'] == [1, 8, 7790]
    assert measured['module_trace']['decoder'] == [1, 256, lengths[-1]]
    specs = [
        ('resnet_vision', 'ResNetVisionPose', '__init__'),
        ('resnet_vision', 'ResNetVisionPose', 'forward'),
        ('vit_vision', 'VisionViTPose', '__init__'),
        ('vit_vision', 'VisionViTPose', 'forward'),
        ('wilor_vit', 'WiLoRViTPose', '__init__'),
        ('wilor_vit', 'WiLoRViTPose', '_extract_feature'),
        ('mid_fusion', 'MidFusionPoseFormer', '__init__'),
        ('mid_fusion', 'MidFusionPoseFormer', '_forward_center_supervised'),
        ('_pooling', 'TemporalAttentionPool', '__init__'),
        ('_pooling', 'TemporalAttentionPool', 'forward'),
    ]
    regions = []
    for filename, cls, method in specs:
        node = member(trees[filename], cls, method)
        regions.append(dict(file=MODULES + filename + '.py', symbol=cls + '.' + method,
                            start_line=node.lineno, end_line=node.end_lineno,
                            ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest()))
    regions.append(dict(file=MODULES + 'mid_fusion.py', symbol='_init_residual_branch',
                        start_line=init_residual.lineno, end_line=init_residual.end_lineno,
                        ast_sha256=hashlib.sha256(ast.dump(init_residual).encode()).hexdigest()))
    value = dict(
        format=FORMAT, upstream_revision=REVISION, source_files_sha256=source,
        source_regions=regions, source_bundle_sha256=digest(source),
        retained_evidence_files_sha256={n: sha(repo / n) for n in EVIDENCE_FILES},
        vision=dict(
            input_shape=['B', 3, 256, 256], num_center_frames=1,
            heads=heads, output_shape=['B', 22, 1],
            resnet_variants={n: resnet_dims[n] for n in ('resnet18', 'resnet50', 'resnet152')},
            dinov2_variants={n: dict(timm_model=dino_dims[n][0], dimension=dino_dims[n][1])
                             for n in ('vit_small', 'vit_base', 'vit_large')},
            resnet_pool='AdaptiveAvgPool2d((1,1)), flatten',
            dinov2_pool='timm backbone(img) vector; no additional pooling in EgoEMG wrapper',
            wilor=dict(input_height_crop=[32, -32], backbone_input_shape=['B', 3, 192, 256],
                       feature_dimension=1280, pool='last 4D returned map, AdaptiveAvgPool2d((1,1)), flatten')),
        fusion=dict(
            recipe='fusion/fusion_vits_s_center_16ch_wl7790', fusion_mode='center_supervised',
            input_shape=['B', 16, 7790], layout=protocol['layout'],
            window_seconds=7790 / 2000, emg_feature_shape=['B', 256, 146],
            token_length_derivation=lengths,
            token_length_basis='pinned convolution arithmetic; reconciled with prior 8-channel diagnostic, not a new 16-channel forward',
            temporal_attention='Linear256->64; Tanh; Linear64->1; softmax over time; weighted sum',
            vision_feature_dimension=384, validity_gate='any(dim=1) if temporal mask; multiply feature before head and projection',
            vision_projection=[384, 256], concat_shape=['B', 512, 1],
            fusion_projection='Conv1d512->512 k1; GELU; Dropout0.1; Conv1d512->256 k1',
            vision_head='Linear384->512; ReLU; Dropout0.1; Linear512->22',
            residual_head='MLPHead256->512; ReLU; Dropout0.1; Linear512->22',
            residual_last_weight_initialization=dict(distribution='normal', mean=0., std=1e-5),
            residual_last_bias_initialization=0., exact_zero_residual_not_guaranteed=True,
            output_shape=['B', 22, 1], prediction='vision_pose + residual',
            prior_actual_native_construction_parameters=construction['model_parameters'],
            retained_first_full_epoch_proves_trained_branch_handoff=True,
            retained_trained_emg_initializer=dict(completed_epochs=150, selected_epoch=133,
                                                 selection=handoff['emg']['checkpoint_selection']),
            retained_trained_vision_initializer=dict(completed_epochs=124,
                                                    selection=handoff['vision']['checkpoint_selection']),
            full_training_complete=False),
        audit_scope='static pinned-source reconciliation with previously retained native execution evidence',
        newly_constructed_models=0, new_model_forwards=0, datasets_loaded=False,
        checkpoints_loaded=False, optimizer_updates=0, cuda_initialized=False,
        scientific_complete=False, full_paper_complete=False,
        renderer_sha256=sha(__file__),
    )
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain',
                                        '--untracked-files=all'], text=True).strip()
    return value


def render(facts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

    assert facts['format'] == FORMAT and facts['renderer_sha256'] == sha(__file__)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.hashsalt': 'egoemg-vision-fusion-architecture-v1'})
    colors = dict(blue='#e6f0f7', green='#e2eddf', purple='#efe5f4', yellow='#f7eedc')
    records = {}

    def setup(width, height):
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set(xlim=(0, width), ylim=(0, height)); ax.axis('off')
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
        return fig, ax

    def box(ax, x, y, w, h, title, detail, color='blue', size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10',
                                   facecolor=colors[color], edgecolor='#476276', linewidth=1))
        ax.text(x + w / 2, y + h - .19, title, ha='center', va='top', fontsize=11, fontweight='bold')
        ax.text(x + w / 2, y + h / 2 - .16, detail, ha='center', va='center', fontsize=size, linespacing=1.5)

    def arrow(ax, points, color='#476276'):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.3)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                    mutation_scale=14, color=color, linewidth=1.3))

    def save(fig, name):
        paths = [output / (name + suffix) for suffix in ('.png', '.svg')]
        fig.savefig(paths[0], dpi=180, metadata={'Software': 'EgoEMG-Reproduction'})
        fig.savefig(paths[1], metadata={'Date': None, 'Creator': 'EgoEMG-Reproduction'})
        plt.close(fig)
        records[name] = {p.suffix[1:]: sha(p) for p in paths}

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = setup(16, 9)
    ax.text(8, 8.65, 'Figure 5: vision-to-pose paths in the pinned code',
            ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 8.2, 'Seven single-frame baselines; separate backbone features and actual angle-regression heads',
            ha='center', fontsize=11)
    box(ax, .25, 4.0, 2.15, 1.7, 'Center-frame crop',
        'RGB 256 x 256\nImageNet normalized\nB x 3 x 256 x 256', 'yellow', 9.3)
    rows = [
        (6.3, 'ResNet-18 / 50 / 152', 'ImageNet-pretrained\nconvolutional backbone',
         'Global average pool', '18: D = 512\n50 / 152: D = 2,048', 'ReLU'),
        (4.0, 'DINOv2 ViT-S / B / L', 'timm pretrained models\npatch size 14; img_size 256',
         'timm feature vector', 'S: 384; B: 768\nL: 1,024', 'ReLU'),
        (1.7, 'WiLoR ViT', 'Crop 32 px top + bottom\n192 x 256 backbone input\nhand-pose pretrained',
         'Global average pool', 'Last 4D feature map\nD = 1,280', 'GELU'),
    ]
    for y, title, body, pool, detail, activation in rows:
        box(ax, 3.05, y, 3.0, 1.65, title, body, size=9.5)
        box(ax, 6.65, y, 2.7, 1.65, pool, detail, 'green', 10)
        box(ax, 9.95, y, 3.05, 1.65, 'Angle regression head',
            'Linear D -> 512\n' + activation + '; dropout 0.1\nLinear 512 -> 22', 'purple')
        box(ax, 13.6, y, 1.95, 1.65, 'Center pose', '22 angles\nB x 22 x 1', size=10)
        arrow(ax, [(2.53, 4.85), (2.72, 4.85), (2.72, y + .825), (2.92, y + .825)])
        for start, end in ((6.18, 6.52), (9.48, 9.82), (13.13, 13.47)):
            arrow(ax, [(start, y + .825), (end, y + .825)])
    ax.text(.3, .55,
            'DINOv2: the EgoEMG wrapper uses backbone(img) directly; no extra spatial mean pool is added here.\n'
            'The generic heads use ReLU; WiLoR uses GELU. WiLoR\'s fixed-code feature dimension is 1,280.\n'
            'Source-based diagram, not a new model run or a claim that all seven full training experiments have finished.\n'
            'Source: zhenqis123/EgoEMG @ ' + REVISION[:12] + '; source hashes and retained execution evidence in the companion JSON.',
            fontsize=9, va='center', linespacing=1.55)
    save(fig, 'vision-architecture')

    fig, ax = setup(18, 10)
    ax.text(9, 9.65, 'Figure 6: native center-supervised residual fusion',
            ha='center', fontsize=18, fontweight='bold')
    ax.text(9, 9.18, 'ViT-S + EMGFormer-S | fusion_vits_s_center_16ch_wl7790 | fixed official training recipe',
            ha='center', fontsize=11)
    box(ax, .25, 7.0, 2.25, 1.5, 'EMG input',
        'B x 16 x 7,790\n8 source channels in 16 slots\n3.895 s at 2 kHz', 'yellow', 9.5)
    box(ax, 3.1, 7.0, 3.0, 1.5, 'TDS + Transformer',
        '2 TDS stages; global SE\n3-layer noncausal RoPE\nB x 256 x 146', size=9.7)
    box(ax, 6.7, 7.0, 3.0, 1.5, 'Temporal attention pool',
        '256 -> 64 -> 1; Tanh\nsoftmax over 146 tokens\nweighted sum: B x 256', 'green', 9.7)
    arrow(ax, [(2.63, 7.75), (2.97, 7.75)])
    arrow(ax, [(6.23, 7.75), (6.57, 7.75)])
    box(ax, .25, 3.7, 2.25, 1.5, 'Center-frame crop', 'RGB 256 x 256\nImageNet normalized\nB x 3 x 256 x 256', 'yellow', 9.5)
    box(ax, 3.1, 3.7, 2.4, 1.5, 'DINOv2 ViT-S', 'timm vector output\nB x 384', size=10)
    box(ax, 6.1, 3.7, 2.1, 1.5, 'Validity gate', 'f_v = m_v * f_v\nbefore BOTH branches', 'green', 9.5)
    arrow(ax, [(2.63, 4.45), (2.97, 4.45)])
    arrow(ax, [(5.63, 4.45), (5.97, 4.45)])
    ax.text(7.15, 5.55, 'm_v: vision_valid_mask\n(any over time if needed)', ha='center', fontsize=9)
    arrow(ax, [(7.15, 5.5), (7.15, 5.33)])
    box(ax, 8.85, 4.7, 1.75, 1.25, 'Vision projection', 'Linear 384 -> 256\nB x 256', size=9)
    box(ax, 10.65, 6.85, 1.65, 1.35, 'Concatenate', 'B x 512\nthen add T = 1', 'green', 9)
    arrow(ax, [(9.83, 7.75), (10.52, 7.75)])
    arrow(ax, [(8.33, 4.45), (8.56, 4.45), (8.56, 5.3), (8.72, 5.3)])
    arrow(ax, [(10.73, 5.3), (11.48, 5.3), (11.48, 6.72)])
    box(ax, 12.9, 6.85, 2.15, 1.35, 'Fusion projection', 'Conv1d 512 -> 512, k1\nGELU; dropout 0.1\nConv1d 512 -> 256, k1', size=8.8)
    box(ax, 15.65, 6.85, 2.05, 1.35, 'Residual head', '256 -> 512 -> 22\nReLU; dropout 0.1\nB x 22 x 1', 'purple', 9)
    arrow(ax, [(12.43, 7.52), (12.77, 7.52)])
    arrow(ax, [(15.18, 7.52), (15.52, 7.52)])
    box(ax, 9.0, 2.65, 3.2, 1.4, 'Vision pose head',
        'Linear 384 -> 512 -> 22\nReLU; dropout 0.1\ny_v: B x 22 x 1', 'purple', 10)
    arrow(ax, [(8.33, 4.45), (8.57, 4.45), (8.57, 3.35), (8.87, 3.35)])
    ax.add_patch(Circle((16.65, 3.35), .32, facecolor=colors['green'], edgecolor='#476276'))
    ax.text(16.65, 3.35, '+', ha='center', va='center', fontsize=20)
    arrow(ax, [(12.33, 3.35), (16.2, 3.35)])
    ax.text(13.7, 3.59, 'y_v', fontsize=11)
    arrow(ax, [(16.65, 6.72), (16.65, 3.8)])
    ax.text(16.85, 4.75, 'delta', fontsize=11)
    box(ax, 14.8, 1.35, 2.9, 1.2, 'Center-frame prediction', 'y = y_v + delta\nB x 22 x 1', 'green', 10)
    arrow(ax, [(16.65, 2.92), (16.65, 2.68)])
    box(ax, .25, 1.2, 7.95, 1.5, 'Native behavior that matters for reproduction',
        'Residual last weight ~ Normal(0, 1e-5); bias = 0: approximately vision-only at initialization.\n'
        'Invalid vision zeros the backbone feature; biased heads can still predict nonzero angles.\n'
        'Only the center target is supervised; this diagram describes the native ViT-S + S path.', 'yellow', 9.5)
    ax.text(.3, .35,
            '146 tokens: derived from the fixed convolution kernels/strides and reconciled with the prior 8-channel shape audit; no new forward here.\n'
            'Native construction and trained-branch handoff have separate retained receipts. Full fusion training is still in progress; no new benchmark.\n'
            'Source: zhenqis123/EgoEMG @ ' + REVISION[:12] + '; the earlier legacy RN18 + S protocol is documented separately.',
            fontsize=9, va='center', linespacing=1.55)
    save(fig, 'fusion-architecture')
    return dict(format='egoemg_vision_fusion_architecture_render_v1',
                facts_sha256=digest(facts), renderer_sha256=sha(__file__), outputs=records,
                python=platform.python_version(),
                dependencies={n: importlib.metadata.version(n) for n in ('matplotlib', 'numpy', 'pillow')},
                new_model_forwards=0, optimizer_updates=0, full_paper_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('audit')
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('render')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        facts = audit(args.upstream, args.repo.resolve())
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(facts, sort_keys=True, indent=2) + '\n')
        print(json.dumps(dict(facts_sha256=sha(args.output), upstream_revision=REVISION, new_model_forwards=0)))
    else:
        facts = json.loads(args.input.read_text())
        receipt = render(facts, args.output_dir)
        (args.output_dir / 'vision-fusion-architecture-render.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
