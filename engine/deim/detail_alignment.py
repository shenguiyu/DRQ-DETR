import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailGateAlignmentLoss(nn.Module):
    """Weak small-object detail alignment for routing gates.

    The loss only pulls selected gate responses up around small objects. It does
    not apply hard background suppression, which keeps medium/large-object
    features from being over-regularized.
    """

    def __init__(
        self,
        small_area=32 * 32,
        small_area_ratio=0.0,
        box_expand=0.15,
        inner_shrink=0.15,
        target_mode="ring",
        dice_weight=0.1,
        smooth_kernel=3,
        warmup_epoch=4,
        eps=1e-6,
    ):
        super().__init__()
        self.small_area = float(small_area)
        self.small_area_ratio = float(small_area_ratio)
        self.box_expand = float(box_expand)
        self.inner_shrink = float(inner_shrink)
        self.target_mode = target_mode
        self.dice_weight = float(dice_weight)
        self.smooth_kernel = int(smooth_kernel)
        self.warmup_epoch = int(warmup_epoch)
        self.eps = eps

    def forward(self, dga_maps, targets, epoch=0):
        if epoch is not None and int(epoch) < self.warmup_epoch:
            return self._zero_from_targets(targets)
        if not dga_maps:
            return self._zero_from_targets(targets)

        losses = []
        for item in dga_maps:
            response = self._as_response(item)
            if response is None:
                continue

            target = self._build_mask(targets, response.shape[-2:], response.device, response.dtype)
            if target.sum() <= 0:
                losses.append(response.sum() * 0.0)
                continue

            response = response.clamp(self.eps, 1.0 - self.eps)
            pos = target > 0.05
            if pos.any():
                pos_weight = target[pos].detach()
                pos_loss = F.binary_cross_entropy(
                    response[pos], torch.ones_like(response[pos]), reduction="none"
                )
                pos_loss = (pos_loss * pos_weight).sum() / pos_weight.sum().clamp_min(self.eps)
            else:
                pos_loss = response.sum() * 0.0

            if self.dice_weight > 0:
                inter = (response * target).sum(dim=(1, 2, 3))
                denom = response.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
                dice_loss = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
                pos_loss = pos_loss + self.dice_weight * dice_loss.mean()

            losses.append(pos_loss)

        if not losses:
            return self._zero_from_targets(targets)
        return torch.stack(losses).mean()

    def _as_response(self, item):
        response = item.get("map", item.get("response", None)) if isinstance(item, dict) else item
        if response is None:
            return None
        if response.dim() == 3:
            response = response.unsqueeze(1)
        if response.dim() != 4:
            return None
        if response.shape[1] != 1:
            response = response.mean(dim=1, keepdim=True)
        return response.float()

    def _build_mask(self, targets, map_hw, device, dtype):
        map_h, map_w = map_hw
        masks = torch.zeros((len(targets), 1, map_h, map_w), device=device, dtype=dtype)

        for batch_idx, target in enumerate(targets):
            boxes = target.get("boxes", None)
            if boxes is None or boxes.numel() == 0:
                continue

            boxes = boxes.detach().to(device=device, dtype=dtype).clamp(0.0, 1.0)
            boxes = boxes[self._small_object_selector(boxes, target, device, dtype)]
            if boxes.numel() == 0:
                continue

            xyxy = self._cxcywh_to_xyxy(boxes)
            xyxy = xyxy * xyxy.new_tensor([map_w, map_h, map_w, map_h])
            x1, y1, x2, y2 = self._expanded_box(xyxy, map_w, map_h)
            outer = torch.zeros((1, map_h, map_w), device=device, dtype=dtype)
            inner = torch.zeros_like(outer)

            for idx, (xa, ya, xb, yb) in enumerate(zip(x1, y1, x2, y2)):
                if xb <= xa or yb <= ya:
                    continue
                outer[:, ya:yb, xa:xb] = 1.0
                if self.target_mode == "ring":
                    ixa, iya, ixb, iyb = self._inner_box(xyxy[idx], map_w, map_h)
                    if ixb > ixa and iyb > iya:
                        inner[:, iya:iyb, ixa:ixb] = 1.0

            if self.target_mode == "ring":
                masks[batch_idx] = (outer - inner).clamp(0.0, 1.0)
            else:
                masks[batch_idx] = outer

        if self.smooth_kernel > 1:
            pad = self.smooth_kernel // 2
            masks = F.max_pool2d(masks, self.smooth_kernel, stride=1, padding=pad)
            masks = F.avg_pool2d(masks, self.smooth_kernel, stride=1, padding=pad).clamp(0.0, 1.0)

        return masks

    def _expanded_box(self, xyxy, map_w, map_h):
        box_w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1.0)
        box_h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1.0)
        expand_x = box_w * self.box_expand
        expand_y = box_h * self.box_expand

        x1 = torch.floor(xyxy[:, 0] - expand_x).clamp(0, map_w - 1).long()
        y1 = torch.floor(xyxy[:, 1] - expand_y).clamp(0, map_h - 1).long()
        x2 = torch.ceil(xyxy[:, 2] + expand_x).clamp(1, map_w).long()
        y2 = torch.ceil(xyxy[:, 3] + expand_y).clamp(1, map_h).long()
        return x1, y1, x2, y2

    def _inner_box(self, xyxy, map_w, map_h):
        box_w = (xyxy[2] - xyxy[0]).clamp(min=1.0)
        box_h = (xyxy[3] - xyxy[1]).clamp(min=1.0)
        shrink_x = box_w * self.inner_shrink
        shrink_y = box_h * self.inner_shrink
        x1 = int(torch.floor(xyxy[0] + shrink_x).clamp(0, map_w - 1).item())
        y1 = int(torch.floor(xyxy[1] + shrink_y).clamp(0, map_h - 1).item())
        x2 = int(torch.ceil(xyxy[2] - shrink_x).clamp(1, map_w).item())
        y2 = int(torch.ceil(xyxy[3] - shrink_y).clamp(1, map_h).item())
        return x1, y1, x2, y2

    def _small_object_selector(self, boxes, target, device, dtype):
        image_h, image_w = self._target_image_size(target, device, dtype)
        pixel_area = boxes[:, 2] * boxes[:, 3] * image_h * image_w
        small = pixel_area <= self.small_area
        if self.small_area_ratio > 0:
            small = small | ((boxes[:, 2] * boxes[:, 3]) <= self.small_area_ratio)
        return small

    def _target_image_size(self, target, device, dtype):
        size = target.get("size", target.get("orig_size", None))
        if size is None:
            return (
                torch.tensor(640.0, device=device, dtype=dtype),
                torch.tensor(640.0, device=device, dtype=dtype),
            )
        size = size.to(device=device, dtype=dtype)
        return size[0].clamp(min=1.0), size[1].clamp(min=1.0)

    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        half_w, half_h = w * 0.5, h * 0.5
        return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)

    @staticmethod
    def _zero_from_targets(targets):
        if targets and "labels" in targets[0]:
            return targets[0]["labels"].float().sum() * 0.0
        return torch.tensor(0.0)
