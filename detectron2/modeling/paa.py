import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from .matcher import Matcher
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import sklearn.mixture as skm


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class PAALossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = sigmoid_focal_loss_jit
        self.focal_gamma = cfg.MODEL.PAA.LOSS_GAMMA
        self.focal_alpha = cfg.MODEL.PAA.LOSS_ALPHA
        self.iou_pred_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(
            cfg.MODEL.PAA.IOU_THRESHOLDS,
            cfg.MODEL.PAA.IOU_LABELS,
            allow_low_quality_matches=True,
        )
        self.box_coder = box_coder

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight)
        else:
            assert losses.numel() != 0
            return losses

    def prepare_iou_based_targets(self, targets, anchors):
        """Compute IoU-based targets"""

        cls_labels = []
        reg_targets = []
        matched_idx_all = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = Boxes.cat(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
            matched_idxs = self.matcher(match_quality_matrix)
            targets_per_im = targets_per_im.copy_with_fields(['labels'])
            matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

            cls_labels_per_im = matched_targets.get_field("labels")
            cls_labels_per_im = cls_labels_per_im.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            cls_labels_per_im[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            cls_labels_per_im[inds_to_discard] = -1

            matched_gts = matched_targets.bbox
            matched_idx_all.append(matched_idxs.view(1, -1))

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets, matched_idx_all

    def compute_paa(self, targets, target_labels, anchors, labels_all, loss_all, matched_idx_all):
        """
            criteria: 'PAA' or 'GMM'

        Args:
            labels_all (batch_size x num_anchors): assigned labels
            loss_all (batch_size x numanchors): calculated loss
        """
        cls_labels = []
        reg_targets = []
        matched_gts = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i].tensor
            bboxes_per_im = targets_per_im
            labels_per_im = target_labels[im_i]
            anchors_per_im = Boxes.cat(anchors[im_i]).tensor
            num_gt = bboxes_per_im.shape[0]

            labels = labels_all[im_i]
            loss = loss_all[im_i]
            matched_idx = matched_idx_all[im_i]
            assert labels.shape == matched_idx.shape

            num_anchors_per_level = [len(anchors_per_level)
                for anchors_per_level in anchors[im_i]]

            # select candidates based on IoUs between anchors and GTs
            candidate_idxs = []
            for gt in range(num_gt):
                candidate_idxs_per_gt = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors[im_i]):
                    end_idx = star_idx + num_anchors_per_level[level]
                    loss_per_level = loss[star_idx:end_idx]
                    labels_per_level = labels[star_idx:end_idx]
                    matched_idx_per_level = matched_idx[star_idx:end_idx]
                    match_idx = ((matched_idx_per_level == gt) &(labels_per_level > 0)).nonzero()[:, 0]
                    if match_idx.numel() > 0:
                        _, topk_idxs = loss_per_level[match_idx].topk(
                            min(match_idx.numel(), self.cfg.MODEL.PAA.TOPK), largest=False)
                        topk_idxs_per_level_per_gt = match_idx[topk_idxs]
                        candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)
                    star_idx = end_idx
                if candidate_idxs_per_gt:
                    candidate_idxs.append(torch.cat(candidate_idxs_per_gt))
                else:
                    candidate_idxs.append(None)

            # fit 2-mode GMM per GT box
            n_labels = anchors_per_im.shape[0]
            cls_labels_per_im = torch.zeros(n_labels, dtype=torch.long).cuda()
            matched_gts_per_im = torch.zeros_like(anchors_per_im)
            fg_inds = matched_idx >= 0
            matched_gts_per_im[fg_inds] = bboxes_per_im[matched_idx[fg_inds]]
            is_grey = None
            for gt in range(num_gt):
                if candidate_idxs[gt] is not None:
                    if candidate_idxs[gt].numel() > 1:
                        candidate_loss = loss[candidate_idxs[gt]]
                        candidate_loss, inds = candidate_loss.sort()
                        candidate_loss = candidate_loss.view(-1, 1).cpu().numpy()
                        min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
                        means_init=[[min_loss], [max_loss]]
                        weights_init = [0.5, 0.5]
                        precisions_init=[[[1.0]], [[1.0]]]
                        gmm = skm.GaussianMixture(2,
                                                  weights_init=weights_init,
                                                  means_init=means_init,
                                                  precisions_init=precisions_init)
                        gmm.fit(candidate_loss)
                        components = gmm.predict(candidate_loss)
                        scores = gmm.score_samples(candidate_loss)
                        components = torch.from_numpy(components).to("cuda")
                        scores = torch.from_numpy(scores).to("cuda")
                        fgs = components == 0
                        bgs = components == 1
                        if fgs.nonzero().numel() > 0:
                            fg_max_score = scores[fgs].max().item()
                            fg_max_idx = (fgs & (scores == fg_max_score)).nonzero().min()
                            is_neg = inds[fgs | bgs]
                            is_pos = inds[:fg_max_idx+1]
                        else:
                            # just treat all samples as positive for high recall.
                            is_pos = inds
                            is_neg = is_grey = None
                    else:
                        is_pos = 0
                        is_neg = None
                        is_grey = None
                    if is_grey is not None:
                        grey_idx = candidate_idxs[gt][is_grey]
                        cls_labels_per_im[grey_idx] = -1
                    if is_neg is not None:
                        neg_idx = candidate_idxs[gt][is_neg]
                        cls_labels_per_im[neg_idx] = 0
                    pos_idx = candidate_idxs[gt][is_pos]
                    cls_labels_per_im[pos_idx] = labels_per_im[gt].view(-1, 1)
                    matched_gts_per_im[pos_idx] = bboxes_per_im[gt].view(-1, 4)

            reg_targets_per_im = self.box_coder.get_deltas(anchors_per_im, matched_gts_per_im)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            matched_gts.append(matched_gts_per_im)

        return cls_labels, reg_targets, matched_gts
