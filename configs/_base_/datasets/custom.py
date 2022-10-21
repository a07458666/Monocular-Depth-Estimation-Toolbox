# dataset settings Only for test
dataset_type = 'CustomDepthDataset'
data_root = './dataset/endoscopy/'
depth_max = 16
depth_min = 1e-3
depth_scale = 65535 / depth_max

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=180.),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(480, 480)),
    dict(type='ColorAug', prob=1, gamma_range=[0.7, 1.3], brightness_range=[0.5, 1.5], color_range=[0.7, 1.3]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(0, 0),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root + 'train',
        depth_scale=depth_scale,
        pipeline=train_pipeline,
        test_mode=False,
        min_depth=depth_min,
        max_depth=depth_max),
    val=dict(
        type=dataset_type,
        data_root=data_root + 'val',
        depth_scale=depth_scale,
        pipeline=test_pipeline,
        min_depth=depth_min,
        max_depth=depth_max,
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'val',
        depth_scale=depth_scale,
        pipeline=test_pipeline,
        test_mode=False,
        min_depth=depth_min,
        max_depth=depth_max))


