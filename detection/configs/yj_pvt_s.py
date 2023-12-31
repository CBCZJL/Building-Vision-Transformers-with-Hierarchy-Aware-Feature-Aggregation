_base_ = [
    '../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(
    pretrained='',
    backbone=dict(
        type='pvt_small',
        style='pytorch'),
    neck=dict(
        type='MTA',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        # add_extra_convs='on_input',
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
        num_outs=5,
        use_sr_layer=False,
    ))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
