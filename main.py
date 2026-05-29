from ultralytics import YOLO
try:
    import thop
except ModuleNotFoundError:
    thop = None
from datetime import datetime
from pathlib import Path
import shutil
import random
import numpy as np
import torch

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*adaptive_max_pool2d_backward_cuda does not have a deterministic implementation.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*cumsum_cuda_kernel does not have a deterministic implementation.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*grid_sampler_2d_backward_cuda does not have a deterministic implementation.*",
)


def monitor_raamamba_health(trainer):
    epoch = trainer.epoch + 1
    if not 5 <= epoch <= 10:
        return

    modules = [
        (name, module)
        for name, module in trainer.model.named_modules()
        if module.__class__.__name__ in {
            "ReliabilityAwareAlignedMambaFusion",
            "StabilizedRAAMambaFusion",
            "GeometryPreservingRAAMambaFusion",
            "BoostedGeometryRAAMambaFusion",
            "ConsistencyGuidedRAAMambaFusion",
            "ResidualConsistencyMambaFusion",
            "ScheduledBoostedGeometryRAAMambaFusion",
        }
    ]
    if not modules:
        return

    print(f"\n[RAA-Mamba health check] epoch={epoch}")
    for name, module in modules:
        gate = module.last_fusion_gate_mean
        reliability = module.last_reliability_mean
        theta = module.last_theta_mean
        delta = module.last_theta_delta_abs_mean

        gate_text = "None" if gate is None else ", ".join(f"{v:.4f}" for v in gate.tolist())
        reliability_text = "None" if reliability is None else ", ".join(f"{v:.4f}" for v in reliability.tolist())
        theta_text = "None" if theta is None else ", ".join(f"{v:.4f}" for v in theta.flatten().tolist())
        delta_text = "None" if delta is None else f"{float(delta):.6f}"
        detail_gate = getattr(module, "last_detail_gate_mean", None)
        consistency = getattr(module, "last_consistency_mean", None)
        schedule = getattr(module, "last_schedule_progress", None)
        detail_text = "" if detail_gate is None else f" detail_gate={float(detail_gate):.4f}"
        consistency_text = "" if consistency is None else f" consistency={float(consistency):.4f}"
        schedule_text = "" if schedule is None else f" schedule={float(schedule):.3f}"
        print(
            f"  {name}({module.stage}) fusion_gate[local,global,fft]=[{gate_text}] "
            f"reliability[vis,ir]=[{reliability_text}] theta_mean=[{theta_text}] "
            f"theta_delta_abs={delta_text}{detail_text}{consistency_text}{schedule_text}"
        )

        if gate is not None and gate[0] > 0.85:
            print(
                f"  WARNING: {name} local gate is {float(gate[0]):.4f} > 0.85. "
                "Mamba/FFT may be marginalized; consider switching to hard residual fusion."
            )


def archive_training_results(save_dir, archive_root="saved_results"):
    save_dir = Path(save_dir)
    archive_dir = Path(archive_root) / save_dir.name

    if archive_dir.exists():
        archive_dir = Path(archive_root) / f"{save_dir.name}_{datetime.now().strftime('%H%M%S')}"

    shutil.copytree(save_dir, archive_dir)
    print(f"\nTraining results archived to: {archive_dir.resolve()}")
    print(f"Best weights: {(archive_dir / 'weights' / 'best.pt').resolve()}")
    print(f"Last weights: {(archive_dir / 'weights' / 'last.pt').resolve()}")


def update_scheduled_mamba_progress(trainer):
    total_epochs = max(int(getattr(trainer, "epochs", 1)), 1)
    progress = min(1.0, max(0.0, (trainer.epoch + 1) / max(total_epochs * 0.35, 1.0)))
    for module in trainer.model.modules():
        if hasattr(module, "set_schedule_progress"):
            module.set_schedule_progress(progress)

if __name__ == '__main__':



    ############## 这是train的代码 ##############
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-64.yaml")  # 初始化模型 纯Concact
    model = YOLO(r"ultralytics/cfg/models/v8/yolov8-c2dff-rcmamba.yaml")  # 从 YAML 随机初始化，训练残差一致性 Mamba 融合模型
    model.add_callback("on_train_epoch_start", update_scheduled_mamba_progress)
    model.add_callback("on_train_epoch_end", monitor_raamamba_health)
    project = Path("runs/detect/train")
    run_name = f"yolov8-c2dff-rcmamba-stable200_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = project / run_name
    #
    model.train(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=8,
                epochs=200, patience=50, project=str(project), name=run_name,
                amp=False,
                deterministic=False,
                workers=4,
                optimizer='SGD',  # Optimizer AdamW SGD
                # cos_lr=True,  # Cosine LR Scheduler
                lr0=0.006, lrf=0.02, warmup_epochs=6.0,
                mosaic=0.4, translate=0.04, scale=0.25, erasing=0.15,
                seed=0, imgsz=640
                )  # 训练 mydata_FLIR.yaml540

    archive_training_results(save_dir)

    # model.train(data=r"ultralytics/cfg/datasets/mydata_FLIR.yaml", batch=8,
    #             epochs=2,project='runs/detect/train', name='yolov8-twoCSP',
    #             amp=False,
    #              workers=4,
    #             optimizer='SGD',  # Optimizer AdamW SGD
    #             # cos_lr=True,  # Cosine LR Scheduler
    #             lr0=0.02,seed=0
    #             )  # 训练 mydata_FLIR.yaml540

    ############## 这是val和predict的代码 ##############
    # VEDAI数据集
    # model = YOLO(r"E:\MUL\weights\C2DFF\vedai\C2DFF_VEDAI.pt")#我的模型
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # VEDAI数据集
    # model.predict(source=r"G:\KeYan\DATA\DroneVehicle\images\test\00013.jpg", save=True,visualize=True)  #   检测一个模态就会自动检测下一个模态 images可见光

    # 其他数据集
    # model = YOLO(r"E:\MUL\weights\C2DFF\FLIR\C2DFF_FLIR.pt")#我的模型
    # model.val(data=r"ultralytics/cfg/datasets/mydata_FLIR.yaml", batch=1, save_json=True, save_txt=False)  # VEDAI数据集

    # # model.predict(source=r"E:\MUL\ultralytics\Datasets\VEDAI_Mine\images\val", save=True)  #   检测一个模态就会自动检测下一个模态
    # model.predict(source=r"G:\KeYan\DATA\VEDAI_1024\images\test", save=True)  #   检测一个模态就会自动检测下一个模态
    # model.predict(source=r"E:/MUL/ultralytics/Datasets/FLIR_align_3class/images/test", save=True)  #   检测一个模态就会自动检测下一个模态
