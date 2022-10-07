import json
import numpy as np
import matplotlib.pyplot as plt
from transformers.manager import PolyPrediction

def read_log(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    real_memory = None
    predict_memory = None
    for line in lines:
        if "real_memory" in line:
            line = line[line.find("{"):-1]
            real_memory = json.loads(line)
        elif "predict_memory" in line:
            line = line[line.find("{"):-1]
            predict_memory = json.loads(line)
    assert real_memory is not None
    assert predict_memory is not None
    return real_memory, predict_memory


def plot_line(real_memory, predict_memory):
    train_num = len(real_memory['x']) - len(predict_memory['x'])
    
    # plot real memory
    plt.scatter(real_memory['x'][train_num:], real_memory['y'][train_num:], label="real memory", s=15)
    
    # plot warmup point
    plt.scatter(real_memory['x'][:train_num], real_memory['y'][:train_num], label="warmup", s=15)
    
    # plot predict_memory
    # plt.scatter(predict_memory['x'], predict_memory['y'], label="predict memory", c="blue", marker="^")
    
    # plot fit line
    x = real_memory['x'][:]
    predict_func = PolyPrediction(real_memory['x'][:train_num], real_memory['y'][:train_num])
    x.sort()
    y = predict_func(x)
    plt.plot(x, y, label="predict line")


def convert_func(memory):
    return {
        'x': memory['seq_length'][:-1],
        'y': np.array(memory['memory'][:-1]) / (1024 ** 3),
    }

def poly_fit(real_memory):
    train_num = 30
    x = real_memory['x'][:train_num]
    y = real_memory['y'][:train_num]
    predict_func = PolyPrediction(x, y)
    return {
        'x': real_memory['x'][train_num:],
        'y': predict_func(real_memory['x'][train_num:]),
    }


def plot_predict(real_memory, _, title):
    real_memory = convert_func(real_memory)
    # predict_memory = convert_func(predict_memory)
    predict_memory = poly_fit(real_memory)
    
    
    fig = plt.figure()
    plot_line(real_memory, predict_memory)
    plt.title(title)
    plt.xlabel("seq length")
    plt.ylabel("increase memory / GB")
    plt.legend()
    plt.savefig(f"./fig/predict/{title}.png")
    plt.close(fig)


def plot_diff_line(real_memory, predict_memory):
    train_num = len(real_memory['x']) - len(predict_memory['x'])
    tmp = [(real_memory['x'][i + train_num], real_memory['y'][i + train_num] - predict_memory['y'][i]) for i in range(len(predict_memory['x']))]
    tmp.sort(key=lambda x: x[0])
    x = [t[0] for t in tmp]
    y = [t[1] for t in tmp]
    plt.plot(x, y)


def plot_diff(real_memory, _, title):
    real_memory = convert_func(real_memory)
    # predict_memory = convert_func(predict_memory)
    predict_memory = poly_fit(real_memory)
    
    fig = plt.figure()
    plot_diff_line(real_memory, predict_memory)
    plt.title(title + " diff")
    plt.xlabel("seq length")
    plt.ylabel("memory / GB")
    plt.savefig(f"./fig/predict/{title}-diff.png")
    plt.close(fig)

def plot_predict_batch():
    filename_map = {
        "log/predict/log.predict_multi-choice": "multiple-choice",
        "log/predict/log.predict_multi-choice-full": "multiple-choice-full",
    }
    for filename, title in filename_map.items():
        real_memory, predict_memory = read_log(filename)
        
        plot_predict(real_memory['model'], predict_memory['model'], title)
        plot_diff(real_memory['model'], predict_memory['model'], title)
        
        if "encoder" in real_memory:
            real_memory['encoder']['seq_length'] = real_memory['model']['seq_length']
            plot_predict(real_memory['encoder'], None, title + "-encoder")
            plot_diff(real_memory['encoder'], None, title + "-encoder")


if __name__ == '__main__':
    plot_predict_batch()
