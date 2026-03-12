from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def mask_to_box(mask: torch.Tensor) -> torch.Tensor | None:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0 or xs.numel() == 0:
        return None

    return torch.tensor(
        [
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        ],
        dtype=torch.float32,
    )


def prepare_image(image: torch.Tensor, sam_model, resize_transform: ResizeLongestSide, device: torch.device):
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    resized_image = resize_transform.apply_image(image_np)
    resized_size = resized_image.shape[:2]

    resized_tensor = torch.as_tensor(resized_image, device=device)
    resized_tensor = resized_tensor.permute(2, 0, 1).contiguous()
    input_image = sam_model.preprocess(resized_tensor.unsqueeze(0))

    return input_image, image_np.shape[:2], resized_size


def apply_augmentations(image: torch.Tensor, gt_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        image = TF.hflip(image)
        gt_masks = TF.hflip(gt_masks.unsqueeze(1).float()).squeeze(1) > 0.5

    if torch.rand(1).item() < 0.8:
        brightness = float(torch.empty(1).uniform_(0.9, 1.1).item())
        contrast = float(torch.empty(1).uniform_(0.9, 1.1).item())
        saturation = float(torch.empty(1).uniform_(0.9, 1.1).item())
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)

    if torch.rand(1).item() < 0.3:
        angle = float(torch.empty(1).uniform_(-7.0, 7.0).item())
        translate = [0, 0]
        scale = float(torch.empty(1).uniform_(0.95, 1.05).item())
        shear = [0.0, 0.0]
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        gt_masks = TF.affine(
            gt_masks.unsqueeze(1).float(),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        ).squeeze(1) > 0.5

    image = image.clamp(0.0, 1.0)
    return image, gt_masks


def overlay_mask(ax, mask: np.ndarray, color: tuple[float, float, float, float]):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)


def draw_box(ax, box: np.ndarray, color: str = "lime"):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=color,
            facecolor=(0, 0, 0, 0),
            linewidth=2,
        )
    )


def save_epoch_visualization(
    model,
    loader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
):
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            for image, target in zip(images, targets):
                gt_masks = target["masks"]
                if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
                    continue

                pred_binary_masks = []
                gt_boxes = []
                image_device = image.to(device)
                input_image, original_size, resized_size = prepare_image(image_device, model, resize_transform, device)
                image_embeddings = model.image_encoder(input_image)

                for mask in gt_masks:
                    box = mask_to_box(mask)
                    if box is None:
                        continue

                    gt_boxes.append(box.cpu().numpy())
                    resized_box = resize_transform.apply_boxes_torch(box.unsqueeze(0), original_size).to(device)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=resized_box,
                        masks=None,
                    )
                    low_res_masks, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_masks = model.postprocess_masks(
                        low_res_masks,
                        input_size=resized_size,
                        original_size=original_size,
                    )
                    pred_binary_masks.append((torch.sigmoid(pred_masks[:, 0]) > 0.5).cpu().numpy()[0])

                if not pred_binary_masks:
                    continue

                image_np = image.permute(1, 2, 0).cpu().numpy()
                gt_masks_np = gt_masks.cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(image_np)
                axes[0].set_title("GT Masks")
                axes[0].axis("off")
                for mask in gt_masks_np:
                    overlay_mask(axes[0], mask, (0.12, 0.56, 1.0, 0.45))
                for box in gt_boxes:
                    draw_box(axes[0], box, color="lime")

                axes[1].imshow(image_np)
                axes[1].set_title("Pred Masks")
                axes[1].axis("off")
                for mask in pred_binary_masks:
                    overlay_mask(axes[1], mask, (0.13, 0.8, 0.33, 0.45))
                for box in gt_boxes:
                    draw_box(axes[1], box, color="lime")

                fig.tight_layout()
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(vis_dir / f"epoch_{epoch:03d}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                return


def forward_single_image(
    model,
    image: torch.Tensor,
    gt_masks: torch.Tensor,
    device: torch.device,
    resize_transform: ResizeLongestSide,
) -> tuple[torch.Tensor | None, int]:
    if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
        return None, 0

    input_image, original_size, resized_size = prepare_image(image, model, resize_transform, device)

    if any(param.requires_grad for param in model.image_encoder.parameters()):
        image_embeddings = model.image_encoder(input_image)
    else:
        with torch.no_grad():
            image_embeddings = model.image_encoder(input_image)

    total_loss = torch.zeros((), device=device)
    valid_instances = 0

    for mask in gt_masks:
        box = mask_to_box(mask)
        if box is None:
            continue

        resized_box = resize_transform.apply_boxes_torch(box.unsqueeze(0), original_size).to(device)

        if any(param.requires_grad for param in model.prompt_encoder.parameters()):
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=resized_box,
                masks=None,
            )
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=resized_box,
                    masks=None,
                )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = model.postprocess_masks(
            low_res_masks,
            input_size=resized_size,
            original_size=original_size,
        )

        pred_logits = pred_masks[:, 0]
        target_mask = mask.unsqueeze(0).float().to(device)

        mask_loss = segmentation_loss(pred_logits, target_mask)

        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
            intersection = (pred_binary * target_mask).sum(dim=(1, 2))
            union = pred_binary.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))
            target_iou = (2.0 * intersection + 1e-6) / (union + 1e-6)

        iou_loss = F.mse_loss(iou_predictions[:, 0], target_iou)
        total_loss = total_loss + mask_loss + iou_loss
        valid_instances += 1

    if valid_instances == 0:
        return None, 0

    return total_loss / valid_instances, valid_instances


def run_epoch(model, loader, optimizer, device, train_mode: bool, augmentations: bool) -> tuple[float, int]:
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)

    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_instances = 0
    progress = tqdm(loader, desc="train" if train_mode else "val")

    for images, targets in progress:
        batch_loss = torch.zeros((), device=device)
        batch_samples = 0
        batch_instances = 0

        for image, target in zip(images, targets):
            image = image.to(device)
            gt_masks = target["masks"].to(device)

            if train_mode and augmentations and gt_masks.numel() > 0:
                image, gt_masks = apply_augmentations(image, gt_masks)

            sample_loss, sample_instances = forward_single_image(
                model=model,
                image=image,
                gt_masks=gt_masks,
                device=device,
                resize_transform=resize_transform,
            )

            if sample_loss is None:
                continue

            batch_loss = batch_loss + sample_loss
            batch_samples += 1
            batch_instances += sample_instances
            total_loss += sample_loss.item() * sample_instances
            total_instances += sample_instances

        if batch_samples == 0:
            continue

        batch_loss = batch_loss / batch_samples

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

        progress.set_postfix(loss=f"{batch_loss.item():.4f}", instances=batch_instances)

    average_loss = total_loss / total_instances if total_instances > 0 else 0.0
    return average_loss, total_instances


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    output_dir: str | Path,
    augmentations: bool = True,
    visualizations: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_instances = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_mode=True,
            augmentations=augmentations,
        )

        with torch.no_grad():
            val_loss, val_instances = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                train_mode=False,
                augmentations=False,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} ({train_instances} instances) | "
            f"val_loss={val_loss:.4f} ({val_instances} instances)"
        )

        if visualizations:
            save_epoch_visualization(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
            )

        last_checkpoint = output_dir / "sam_task_e_last.pth"
        torch.save(model.state_dict(), last_checkpoint)

        if val_instances > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = output_dir / "sam_task_e_best.pth"
            torch.save(model.state_dict(), best_checkpoint)
            print(f"Saved best checkpoint to {best_checkpoint}")
