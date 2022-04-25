import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='analyse train tensorboard log file')
parser.add_argument('--log-files', type=str, nargs='*', required=True, help='log file path, json format')
parser.add_argument('--names', type=str, nargs='*', required=True, help='train type names')
args = parser.parse_args()


def load_logs():
    log_dict = dict()
    for i in range(len(args.log_files)):
        log_path = args.log_files[i]
        log_name = args.names[i]
        with open(log_path, 'r') as f:
            log_dict[log_name] = list(map(lambda x: x[1:], json.load(f)))
    return log_dict


def smooth_data(x, y, dense=1500):
    from scipy.interpolate import make_interp_spline
    x, y = np.array(x), np.array(y)
    x_new = np.linspace(x.min(), x.max(), dense)
    y_new = make_interp_spline(x, y)(x_new)
    return list(x_new), list(y_new)


def crop_data(x, y, a, b):
    return x[a:b], y[a:b]


def main():
    assert len(args.log_files) == len(args.names), 'log file num must eq name num'
    log_dict = load_logs()
    plt.figure()
    for log_name in log_dict.keys():
        log_data = log_dict[log_name]
        x_list = list(map(lambda x: x[0], log_data))
        y_list = list(map(lambda x: x[1], log_data))
        x_list, y_list = crop_data(x_list, y_list, 122, 128)
        x_list, y_list = smooth_data(x_list, y_list, )
        plt.plot(
            x_list,
            y_list,
            label=log_name,
        )
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
