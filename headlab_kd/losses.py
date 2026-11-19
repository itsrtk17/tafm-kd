import torch
import torch.nn as nn
import torch.nn.functional as F


def raw_output(value):
    if isinstance(value, tuple):
        value = value[1]
    if 'one2many' in value:
        value = value['one2many']
    if not {'boxes', 'scores', 'feats'} <= value.keys():
        raise RuntimeError('Unexpected prediction interface; use inspected Ultralytics 8.4.176.')
    return value


def channels(head):
    return [next(m for m in tower.modules() if isinstance(m, nn.Conv2d)).in_channels for tower in head.cv2]


def decode(raw, head):
    logits = raw['boxes']
    batch, _, count = logits.shape
    bins = torch.arange(head.reg_max, device=logits.device, dtype=logits.dtype).view(1, 1, -1, 1)
    distances = (logits.reshape(batch, 4, head.reg_max, count).softmax(2) * bins).sum(2)
    anchors, strides, sizes = [], [], []
    for feature, stride in zip(raw['feats'], head.stride):
        h, w = feature.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=logits.device), torch.arange(w, device=logits.device), indexing='ij')
        anchors.append(torch.stack((xx, yy), 0).reshape(2, -1).to(logits.dtype) + .5)
        strides.append(torch.ones(1, h*w, device=logits.device, dtype=logits.dtype) * stride)
        sizes.append(h*w)
    anchor = torch.cat(anchors, 1).unsqueeze(0)
    stride = torch.cat(strides, 1).unsqueeze(0)
    boxes = torch.cat(((anchor-distances[:, :2])*stride, (anchor+distances[:, 2:])*stride), 1).transpose(1, 2)
    return boxes, (anchor*stride).transpose(1, 2)[0], sizes


def aligned_giou(first, second):
    intersection = (torch.minimum(first[..., 2:], second[..., 2:]) - torch.maximum(first[..., :2], second[..., :2])).clamp(min=0).prod(-1)
    a = (first[..., 2:]-first[..., :2]).clamp(min=0).prod(-1)
    b = (second[..., 2:]-second[..., :2]).clamp(min=0).prod(-1)
    union = a+b-intersection
    enclosure = (torch.maximum(first[..., 2:], second[..., 2:])-torch.minimum(first[..., :2], second[..., :2])).clamp(min=0).prod(-1)
    return intersection/union.clamp(min=1e-7) - (enclosure-union)/enclosure.clamp(min=1e-7)


@torch.no_grad()
def reliable_mask(boxes, centers, batch, size, threshold=.25):
    """Declared adaptation: GT-containing anchor AND teacher-to-that-GT IoU >= threshold.

    No post-NMS matching. Same grid positions correspond to identical anchors.
    Chunking bounds temporary memory in dense images.
    """
    masks = []
    for b in range(boxes.shape[0]):
        xywh = batch['bboxes'][batch['batch_idx'].long() == b] * size
        gt = torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), 1)
        mask = torch.zeros(len(centers), device=boxes.device, dtype=torch.bool)
        if len(gt):
            gt_area = (gt[:, 2:]-gt[:, :2]).clamp(min=0).prod(-1)
            for start in range(0, len(centers), 512):
                center = centers[start:start+512]
                pred = boxes[b, start:start+512]
                inside = ((center[:, None, :] >= gt[None, :, :2]) & (center[:, None, :] <= gt[None, :, 2:])).all(-1)
                inter = (torch.minimum(pred[:, None, 2:], gt[None, :, 2:])-torch.maximum(pred[:, None, :2], gt[None, :, :2])).clamp(min=0).prod(-1)
                pred_area = (pred[:, 2:]-pred[:, :2]).clamp(min=0).prod(-1)
                iou = inter/(pred_area[:, None]+gt_area[None, :]-inter).clamp(min=1e-7)
                mask[start:start+512] = ((iou >= threshold) & inside).any(-1)
        masks.append(mask)
    return torch.stack(masks)


class CrossingKD(nn.Module):
    """Adapters only: teacher remains external and is never registered here."""

    def __init__(self, student_head, teacher_head):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Conv2d(a, b, 1, bias=False) for a, b in zip(channels(student_head), channels(teacher_head))])

    def forward(self, student_raw, teacher_raw, teacher_head, batch, quality=.25):
        mapped = [adapter(feature) for adapter, feature in zip(self.adapters, student_raw['feats'])]
        # Frozen teacher parameters; graph through its operations MUST remain enabled.
        crossed = teacher_head.forward_head(mapped, box_head=teacher_head.cv2, cls_head=teacher_head.cv3)
        if crossed['scores'].shape != teacher_raw['scores'].shape:
            raise RuntimeError('Teacher/cross prediction grids differ')
        cross_boxes, centers, sizes = decode(crossed, teacher_head)
        teacher_boxes, teacher_centers, teacher_sizes = decode(teacher_raw, teacher_head)
        if sizes != teacher_sizes or not torch.equal(centers, teacher_centers):
            raise RuntimeError('Teacher/cross anchors do not align')
        mask = reliable_mask(teacher_boxes.detach(), centers, batch, batch['img'].shape[-1], quality)
        targets = teacher_raw['scores'].detach().sigmoid()
        cls = F.binary_cross_entropy_with_logits(crossed['scores'], targets, reduction='none') * (crossed['scores'].sigmoid()-targets).abs()  # beta=1
        cls = cls.mean(1)
        reg = (1-aligned_giou(cross_boxes, teacher_boxes.detach())).square()  # gamma=2
        # Equal mean over nonempty feature levels, rather than letting P2's grid dominate.
        cls_terms, reg_terms, counts = [], [], []
        start = 0
        for count in sizes:
            selected = mask[:, start:start+count]
            n = int(selected.sum().item())
            counts.append(n)
            if n:
                cls_terms.append(cls[:, start:start+count][selected].mean())
                reg_terms.append(reg[:, start:start+count][selected].mean())
            start += count
        zero = crossed['scores'].sum()*0 + crossed['boxes'].sum()*0
        cls_loss = torch.stack(cls_terms).mean() if cls_terms else zero
        reg_loss = torch.stack(reg_terms).mean() if reg_terms else zero
        return cls_loss, reg_loss, counts
