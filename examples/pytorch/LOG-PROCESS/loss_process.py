import json
import numpy as np
import matplotlib.pyplot as plt


def is_loss_line(line):
    keys = ['loss', 'learning_rate', 'steps', 'epoch']
    for key in keys:
        if key not in line:
            return False
    return True

def convert_log(origin_log):
    origin_log.sort(key=lambda x: x['steps'])
    return origin_log
    

def read_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    origin_log = []
    for line in lines:
        if is_loss_line(line):
            line = line[:-1].replace("'", '"')
            origin_log.append(json.loads(line))
    origin_log = convert_log(origin_log)
    return origin_log


def plot_loss(loss_dict, title):
    fig = plt.figure()
    regular_result = loss_dict.pop("regular")
    regular_result = [tmp['loss'] for tmp in regular_result]
    regular_result = np.array(regular_result)
    for label, loss_log in loss_dict.items():
        x = [tmp['steps'] for tmp in loss_log]
        y = [tmp['loss'] for tmp in loss_log]
        plt.plot(x, np.array(y) - regular_result, label=label)
    plt.title(title)
    plt.xlabel("steps")
    plt.ylabel("loss diff")
    plt.legend()
    plt.savefig(f"./fig/loss/{title}.png")
    plt.close(fig)


def process_loss_log(filename_prefix, title):
    works = ['regular', 'gc', 'sublinear', 'dc']
    loss_dict = {}
    for work in works:
        filename = filename_prefix + work
        loss_dict[work] = read_log(filename)
    
    plot_loss(loss_dict, title)

def batch_process_loss_log():
    filename_prefix_map = {
        "./log/loss/qa/log.loss_": "qa",
        "./log/loss/multi-choice/log.loss_": "multiple-choice",
        "./log/loss/text-classification/log.loss_qqp_": "text-classification",
    }
    for filename_prefix, title in filename_prefix_map.items():
        process_loss_log(filename_prefix, title)

if __name__ == "__main__":
    batch_process_loss_log()