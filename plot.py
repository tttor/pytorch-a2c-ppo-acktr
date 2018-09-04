#!/usr/bin/env python3
import os
import csv
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

repo_name = 'ikostrikov_pytorch-a2c-ppo-acktr'

def main():
    args = parse_args()

    mon_data = load_monitor_data(args.log_dir)
    plot_learning_curve(mon_data, args.log_dir)

    opt_data = load_opt_data(args.log_dir)
    plot_opt_data('policy loss', opt_data, args.log_dir)
    plot_opt_data('value loss', opt_data, args.log_dir)
    plot_opt_data('entropy', opt_data, args.log_dir)

def plot_opt_data(mode, data, log_dir):
    x, y = data['Updates'], data[mode]
    y_min = min(y)
    if y_min < 0.: y = [yi+abs(y_min) for yi in y]
    fig, ax = plt.subplots()
    plt.semilogy(x, y, '-')
    plt.grid(True)
    plt.xlabel('ith update')
    plt.ylabel(mode.lower()+(': y+'+str(abs(y_min))) if y_min < 0. else '')
    plt.title(' '.join(['PPO', mode.lower(), repo_name]))
    plt.savefig(os.path.join(log_dir,mode.lower().replace(' ','_')+'_'+repo_name+'.png'),dpi=300,format='png',bbox_inches='tight');
    plt.close(fig)

def plot_learning_curve(data, log_dir):
    fig, ax = plt.subplots()
    x, y = data
    plt.plot(x, y, '-')
    ytick_offset = 5
    yticks = np.arange(min(y)-ytick_offset, max(y)+ytick_offset, 10)
    plt.grid(True)
    plt.yticks(yticks)
    plt.xlabel('#steps')
    plt.ylabel('return (undiscounted)')
    plt.title(' '.join(['PPO', 'return', repo_name]))
    plt.savefig(os.path.join(log_dir,'return_'+repo_name+'.png'),dpi=300,format='png',bbox_inches='tight');
    plt.close(fig)

def load_opt_data(log_dir):
    data = []; data_dict = {}
    keys = ['Updates', 'num timesteps', 'entropy', 'value loss', 'policy loss']
    with open(os.path.join(log_dir, 'terminal.txt'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v.replace(k,'')) for k, v in row.items() if k in keys})
    for k in keys:
        data_dict[k] = [datum[k] for datum in data]
    return data_dict

def load_monitor_data(log_dir):
    # monitoring data produced with:
    # /home/tor/ws/poacp/xternal/baselines/baselines/bench/monitor.py
    # L66: epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
    data = [] # one episode info per row
    fpaths = [os.path.join(log_dir, fname) for fname in os.listdir(log_dir) if 'monitor.csv' in fname]
    for fpath in fpaths:
        print(fpath)
        with open(fpath, 'r') as f:
            f.readline()# skip the fist line, eg. #{"t_start": 1531702562.0624273, "env_id": "Reacher-v2"}
            reader = csv.DictReader(f)
            for row in reader:
                data.append({k: float(v) for k, v in row.items()})
    data = sorted(data, key=lambda entry: entry['t'])

    nstep_return_data = []
    nstep = 0 # nstep so far
    for _, datum in enumerate(data):
        nstep_return_data.append((nstep, datum['r']))
        nstep += int(datum['l'])

    xy = torch.tensor(nstep_return_data)
    x, y = torch.chunk(xy, 2, dim=1)
    x = x.squeeze().numpy()
    y = y.squeeze().numpy()

    x, y = smooth_reward_curve(x, y)
    x, y = fix_point(x, y, interval=100)

    return (x, y)

def smooth_reward_curve(x, y):
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/4d95ec364c7303566c6a52fb0a254640e931609d/visualize.py#L18
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]

def fix_point(x, y, interval):
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/4d95ec364c7303566c6a52fb0a254640e931609d/visualize.py#L29
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='root xprmt log dir', type=str, default=None, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main()
