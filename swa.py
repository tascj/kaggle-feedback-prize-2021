import argparse
import os.path as osp

import torch


def average_checkpoints(input_ckpts, output_ckpt):
    assert len(input_ckpts) >= 1
    data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
    swa_n = 1
    for ckpt in input_ckpts[1:]:
        new_data = torch.load(ckpt, map_location='cpu')['state_dict']
        swa_n += 1
        for k, v in new_data.items():
            if v.dtype == torch.int64:
                continue
            data[k] += (new_data[k] - data[k]) / swa_n
    torch.save(dict(state_dict=data), output_ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir')
    parser.add_argument('start_epoch', type=int)
    parser.add_argument('end_epoch', type=int)
    args = parser.parse_args()

    ckpts = [
        osp.join(args.work_dir, f'epoch_{i}.pth')
        for i in range(args.start_epoch, args.end_epoch + 1)
    ]
    average_checkpoints(ckpts, osp.join(args.work_dir, 'swa.pth'))


if __name__ == '__main__':
    main()
