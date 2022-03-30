import os

os.environ['TOKENIZERS_PARALLELISM'] = '0'

import argparse
import logging
import re
import time
import os.path as osp

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from transformers import AutoModelForTokenClassification, AutoTokenizer

import mmcv
from mmcv.utils import get_logger
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import BaseModule, build_runner, build_optimizer, OPTIMIZERS

from transformers.optimization import AdamW

OPTIMIZERS.register_module(name='HuggingFaceAdamW', module=AdamW)


def to_gpu(data):
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep


class TextSpanDetector(BaseModule):
    def __init__(self,
                 arch,
                 num_classes=7,
                 dynamic_positive=False,
                 with_cp=False,
                 local_files_only=True,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.dynamic_positive = dynamic_positive
        self.model = AutoModelForTokenClassification.from_pretrained(
            arch,
            num_labels=1 + 2 + num_classes,
            local_files_only=local_files_only)
        if with_cp:
            self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(
            arch, local_files_only=local_files_only)

        # init bias
        self.model.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        self.model.classifier.bias.data[3:].fill_(
            bias_init_with_prob(1 / self.num_classes))

    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        outputs = self.model(input_ids=data['input_ids'],
                             attention_mask=data['attention_mask'])
        logits = outputs['logits']
        logits = aggregate_tokens_to_words(logits, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]
        return obj_pred, reg_pred, cls_pred

    def predict(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(text_id=data['text_id'],
                    start=to_np(start),
                    end=to_np(end),
                    score=to_np(pr_score),
                    label=to_np(pr_label))

    def train_step(self, data, optimizer, **kwargs):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_target, reg_target, cls_target, pos_loc = self.build_target(
            data['gt_spans'], obj_pred, reg_pred, cls_pred)

        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred,
                                                       cls_pred, obj_target,
                                                       reg_target, cls_target,
                                                       pos_loc)
        loss = obj_loss + reg_loss + cls_loss
        log_vars = dict(
            obj_loss=obj_loss.item(),
            reg_loss=reg_loss.item(),
            cls_loss=cls_loss.item(),
            loss=loss.item(),
        )
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=1)

        return outputs

    def get_losses(self, obj_pred, reg_pred, cls_pred, obj_target, reg_target,
                   cls_target, pos_loc):
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        reg_pred = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        reg_loss = -iou.log().sum() / num_total_samples
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        return obj_loss, reg_loss, cls_loss

    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        # dynamically assign one more positive
        if self.dynamic_positive:
            cls_prob = (obj_pred.sigmoid().unsqueeze(1) *
                        cls_pred.sigmoid()).sqrt()
            for start, end, label in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                cls_cost = F.binary_cross_entropy(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)
                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
            pos_loc = (obj_target == 1).nonzero().flatten()
        return obj_target, reg_target, cls_target, pos_loc


LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')

TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}


class FeedbackDataset(Dataset):
    def __init__(self,
                 csv_file,
                 text_dir,
                 tokenizer,
                 mask_prob=0.0,
                 mask_ratio=0.0,
                 query=None):
        self.df = pd.read_csv(csv_file)
        if query is not None:
            self.df = self.df.query(query).reset_index(drop=True)
        self.samples = list(self.df.groupby('id'))
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

        assert 0 <= mask_prob <= 1
        assert 0 <= mask_ratio <= 1
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]
        text_path = osp.join(self.text_dir, f'{text_id}.txt')

        with open(text_path) as f:
            text = f.read().rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        # word slices of ground truth spans
        gt_spans = []
        for _, row in df.iterrows():
            winds = row['predictionstring'].split()
            start = int(winds[0])
            end = int(winds[-1])
            span_label = TYPE2LABEL[row['discourse_type']]
            gt_spans.append([start, end + 1, span_label])
        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(text=text,
                    text_id=text_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class CustomCollator(object):
    def __init__(self, tokenizer, model):
        self.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (model.config.attention_window
                                     if isinstance(
                                         model.config.attention_window, int)
                                     else max(model.config.attention_window))
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger('feedback-prize', log_file, logging.INFO)
    detector = TextSpanDetector(**cfg.model).cuda()

    optimizer = build_optimizer(detector, cfg.optimizer)
    runner = build_runner(cfg.runner,
                          default_args=dict(model=detector,
                                            batch_processor=None,
                                            optimizer=optimizer,
                                            work_dir=cfg.work_dir,
                                            logger=logger,
                                            meta=None))
    runner.timestamp = timestamp
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None),
                                   custom_hooks_config=cfg.get(
                                       'custom_hooks', None))

    fold = cfg.data.pop('fold')

    train_ds = FeedbackDataset(**cfg.data,
                               tokenizer=detector.tokenizer,
                               query=f'fold != {fold}')
    collate = CustomCollator(detector.tokenizer, detector.model)
    train_dl = DataLoader(train_ds,
                          batch_size=1,
                          shuffle=True,
                          num_workers=2,
                          collate_fn=collate)

    runner.run([train_dl], [('train', 1)])


if __name__ == '__main__':
    main()
