import matplotlib.pyplot as plt
import numpy as np
import json

def cast_key(inputs):
    tmp = {}
    for k, v in inputs.items():
        tmp[int(k)] = v
    return tmp


def read_log(filename):
    print(f"read file {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    shape_count = None
    memory_count = None
    def replace_line(line, target):
        return line[line.find(target) + len(target):]
    for line in lines:
        if "shape_count" in line:
            shape_count = json.loads(replace_line(line, "shape_count=")[:-1])
        elif "memory_count" in line:
            memory_count = json.loads(replace_line(line, "memory_count=")[:-1])
            break
    assert shape_count is not None
    # return cast_key(shape_count), 1
    assert memory_count is not None
    return cast_key(shape_count), cast_key(memory_count)

def get_train_time(filename) -> float:
    """ get epoch time in log

    Args:
        filename (str): log filename

    Returns:
        float: epoch time(min)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    max_time = 0
    for line in lines:
        if "100%" in line:
            time_str = line.split("[")[1].split('<')[0].split(':')
            epoch_time = int(time_str[0]) + int(time_str[1]) / 60
            max_time = max(max_time, epoch_time)
        if "shape_count" in line:
            break
    return max_time


def plot_distribute(shape_count, title="conll2003-batch64"):
    bin = []
    for key, value in shape_count.items():
        bin += [key] * value
    fig = plt.figure()
    plt.hist(bin, bins=30)
    plt.xlabel("seq length")
    plt.ylabel("number")
    plt.title(title)
    plt.savefig(f"fig/dataset/{title}.png")
    plt.close(fig)


def plot_memory(memory_count, title):
    x, y = [], []
    for key, value in memory_count.items():
        x += [key] * len(value)
        y += (np.array(value) / 1024 / 1024).tolist()
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("seq length")
    plt.ylabel("memory used / MB")
    plt.title(title)
    plt.savefig(f"fig/memory-{title}.png")

def plot_memory_line(memory_count, label):
    x, y = [], []
    for key, value in memory_count.items():
        x += [key] * len(value)
        y += (np.array(value) / 1024 / 1024).tolist()
    plt.scatter(x, y, label=label, s=1)

def plot_memory_all():
    # prefix = "../text-classification/train_log/log.train_qqp_dc_"
    # file_list = ["2.05222240", "3.05230043", "4.05230243", "5.05230443", "6.05230642", "7.05230841", "8.05231031"]
    # title = "glue-qqp-bert_base-batch32-dc-23"

    prefix = "../multiple-choice/train_log/log.train_dc_"
    file_list = ["2.05231439", "3.05231509", "4.05231539", "5.05231610", "6.05231640", "7.05231712", "8.05231742"]
    title = "multiple_choice-bert_base-batch16-dc-23"
    plt.figure()
    for filename in file_list:
        _, memory_count = read_log(prefix + filename)
        print(f"dc-{filename.split('.')[0]} training time: {get_train_time(prefix + filename):.2f} min")
        plot_memory_line(memory_count, "dc-" + filename.split('.')[0])
    plt.xlabel("seq length")
    plt.ylabel("memory used / MB")

    plt.title(title)
    plt.legend()
    plt.savefig(f"fig/memory-{title}.png")

def plot_memory_checkpoint():
    prefix = "../multiple-choice/train_log/"
    filename = "log.train.05231809"
    # gc_filename = "log.train_gc.05230717"
    gc_filename = "log.train_prev_cp.04251500"
    dc_filename = "log.train_dc_5.05231610"
    title = "multiple_choice-bert_base-batch16-checkpoint"
    
    plt.figure()
    
    _, memory_count = read_log(prefix + filename)
    print(f"origin training time: {get_train_time(prefix + filename):.2f} min")
    plot_memory_line(memory_count, "regular")
    
    _, memory_count = read_log(prefix + gc_filename)
    print(f"gradient checkpoint training time: {get_train_time(prefix + gc_filename):.2f} min")
    plot_memory_line(memory_count, "gradient checkpoint")
    
    # _, memory_count = read_log(prefix + dc_filename)
    # print(f"gradient checkpoint training time: {get_train_time(prefix + dc_filename):.2f} min")
    # plot_memory_line(memory_count, "dc")
    
    
    plt.xlabel("seq length")
    plt.ylabel("memory used / MB")

    plt.title(title)
    plt.legend()
    plt.savefig(f"fig/gc-{title}.png")


def plot_budget_overhead(prefix, file_list, title, baseline_filename, sublinear_filename, gc_filename):
    pair = []
    for filename in file_list:
        _, memory_count = read_log(prefix + filename)
        memory_budget = int(filename.split(".")[0])
        if len(memory_count) == 0:
            x = memory_budget
        else:
            x = max([max(value) for value in memory_count.values()]) / 1024 / 1024 / 1024 # GB
        y = get_train_time(prefix + filename) # minutes
        pair.append((x, y, memory_budget))
    pair.sort()
    x, y, annotations = list(zip(*pair))

    baseline_time = get_train_time(baseline_filename)
    _, memory_count = read_log(baseline_filename)
    baseline_memory = max([max(value) for value in memory_count.values()]) / 1024 / 1024 / 1024 # GB
    y = np.array(y) / baseline_time
    plt.figure()
    plt.scatter(x, y, label="dynamic checkpoint")
    # plt.plot(x, y)
    for tmp_x, tmp_y, memory_budget in pair:
        plt.annotate(f"{memory_budget} GB", (tmp_x, tmp_y / baseline_time))

    y = [get_train_time(sublinear_filename) / baseline_time]
    _, memory_count = read_log(sublinear_filename)
    x = [max([max(value) for value in memory_count.values()]) / 1024 / 1024 / 1024]
    plt.scatter(x, y, label="sublinear")

    y = [get_train_time(gc_filename) / baseline_time]
    _, memory_count = read_log(gc_filename)
    x = [max([max(value) for value in memory_count.values()]) / 1024 / 1024 / 1024]
    plt.scatter(x, y, label="gradient checkpoint")

    plt.scatter([baseline_memory], [1.0], label="regular")
    
    plt.xlabel("memory budget")
    plt.ylabel("overhead")
    plt.title(title)
    plt.legend()
    plt.savefig(f"fig/overhead-{title}.png")


def plot_budget_overhead_all():
    prefix = "../text-classification/train_log/log.train_qqp_dc_"
    file_list = ["2.05292203", "3.05292344", "4.05301447", "5.05301530", "6.05301609", "7.05301649", "8.05301822"]
    title = "glue-qqp-bert_base-batch32-budget_overhead"
    baseline_filename = "../text-classification/train_log/log.train_qqp.05300133"
    sublinear_filename = "../text-classification/train_log/log.train_qqp_sublinear.05300212"
    gc_filename = "../text-classification/train_log/log.train_qqp_gc.05300305"
    plot_budget_overhead(prefix, file_list, title, baseline_filename, sublinear_filename, gc_filename)
    
    prefix = "../multiple-choice/train_log/log.train_dc_"
    file_list = ["2.05292209", "3.05292323", "4.05301450", "5.05301522", "6.05301551", "7.05301619", "8.05301647"]
    title = "swag-bert_base-batch16-budget_overhead"
    baseline_filename = "../multiple-choice/train_log/log.train.05300156"
    sublinear_filename = "../multiple-choice/train_log/log.train_sublinear.05300301"
    gc_filename = "../multiple-choice/train_log/log.train_gc.05300224"
    plot_budget_overhead(prefix, file_list, title, baseline_filename, sublinear_filename, gc_filename)

if __name__ == "__main__":
    # filename = "../token-classification/train_log/log.train.04171437"
    # shape_count, memory_count = read_log(filename)
    # plot_distribute(shape_count, title="conll2003-batch64")
    # plot_memory(memory_count, title="conll2003-bert_base-batch64")

    # filename = "../question-answering/train_log/log.train_dc_8.05081713"
    # filename = "../text-classification/train_log/log.train_qqp_ds.05152225"
    # shape_count, memory_count = read_log(filename)
    # plot_distribute(shape_count, title="glue-qqp-batch32")
    # plot_memory(memory_count, title="glue-qqp-bert_base-batch32-deepspeed")
    # plot_memory_all()
    plot_memory_checkpoint()
    # plot_budget_overhead_all()
