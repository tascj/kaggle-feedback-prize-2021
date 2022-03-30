model = dict(arch='microsoft/deberta-xlarge',
             with_cp=True,
             dynamic_positive=True)

data = dict(fold=0,
            csv_file='../data/dtrainval.csv',
            text_dir='../data/train',
            mask_prob=0.8,
            mask_ratio=0.3)

runner = dict(type='EpochBasedRunner', max_epochs=6)

start_lr_ratio = 1e-3
lr = 2.5e-5 * start_lr_ratio
batch_size = 16
lr_config = dict(
    policy='Cyclic',
    by_epoch=False,
    target_ratio=(1 / start_lr_ratio, 1),
    cyclic_times=runner['max_epochs'],
    step_ratio_up=0.05,
    anneal_strategy='linear',
    gamma=1,
)
optimizer = dict(
    type='HuggingFaceAdamW',
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
    paramwise_cfg=dict(norm_decay_mult=0.,
                       bias_decay_mult=0.,
                       custom_keys={
                           'classifier': dict(lr_mult=10),
                       }),
)
optimizer_config = dict(type='GradientCumulativeOptimizerHook',
                        cumulative_iters=batch_size,
                        grad_clip=dict(max_norm=1.0, norm_type=2.0))

checkpoint_config = dict(interval=1, save_optimizer=False)
log_config = dict(interval=800, hooks=[dict(type='TextLoggerHook')])
load_from = None
resume_from = None
workflow = [('train', 1)]
