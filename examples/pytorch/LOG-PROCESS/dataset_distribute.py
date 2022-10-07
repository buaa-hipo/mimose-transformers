import itertools
import numpy as np
import matplotlib.pyplot as plt
from log_process import read_log, plot_distribute

def plot_dataset_distribute(filename, title):
    shape_count, _ = read_log(filename)
    plot_distribute(shape_count, title)

def read_shape_count(filename):
    return read_log(filename)[0]

def plot_distribute_line(shape_count, factor, label):
    bin = []
    bins_number = 10
    for in_size, num in shape_count.items():
        bin += [in_size * factor] * num
    bin = np.array(bin)
    max_number = bin.max()
    min_number = bin.min()
    x = [(i + 0.5) * (max_number - min_number) / bins_number  + min_number for i in range(bins_number)]
    y = [(((i + 1) * (max_number - min_number) / bins_number  + min_number) > bin).sum() for i in range(bins_number)]
    for i in range(bins_number - 1, 0, -1):
        y[i] -= y[i - 1]
        y[i] /= len(bin)
    y[0] /= len(bin)
    plt.plot(x, y, label=label)

def plot_dataset_distribute_togather(batch_size_set, dataset_filename, image_name="dataset distribute"):
    fig = plt.figure()
    for dataset_pair, batch_size in itertools.product(dataset_filename.items(), batch_size_set):
        dataset_name, value = dataset_pair
        filename_prefix, factor = value
        # factor *= batch_size
        title = dataset_name + f"-{batch_size}"
        filename = filename_prefix + str(batch_size)
        
        shape_count = read_shape_count(filename)
        plot_distribute_line(shape_count, factor, title)
    plt.xlabel("input size")
    plt.ylabel("frequency")
    plt.legend()
    plt.title(image_name)
    plt.savefig(f"fig/dataset-{image_name.replace(' ', '-')}.png")
    plt.close(fig)
    
def plot_dataset_distribute_figure():
    dataset_filename = {
        "coco": ("./log/log.dataset_mscoco2017_", 3),
    }
    batch_size_set = [1, 16, 128]
    plot_dataset_distribute_togather(batch_size_set, dataset_filename, "coco")
    
    dataset_filename = {
        # "glue-qqp": ("../text-classification/train_log/log.dataset_glue_qqp_", 1),
        "swag": ("../multiple-choice/train_log/log.dataset_swag_", 4),
        # "squad": ("../question-answering/train_log/log.dataset_squad_", 1),
    }
    batch_size_set = [1, 16, 128]
    plot_dataset_distribute_togather(batch_size_set, dataset_filename, "NLP")

def plot_dataset_distribute_split_figure():
    dataset_filename = {
        "glue-qqp": ("../text-classification/train_log/log.dataset_glue_qqp_"),
        "glue-cola": ("../text-classification/train_log/log.dataset_glue_cola_"),
        "glue-sst2": ("../text-classification/train_log/log.dataset_glue_sst2_"),
        "glue-mrpc": ("../text-classification/train_log/log.dataset_glue_mrpc_"),
        "glue-stsb": ("../text-classification/train_log/log.dataset_glue_stsb_"),
        "glue-mnli": ("../text-classification/train_log/log.dataset_glue_mnli_"),
        "glue-rte": ("../text-classification/train_log/log.dataset_glue_rte_"),
        "glue-wnli": ("../text-classification/train_log/log.dataset_glue_wnli_"),
        "swag": ("../multiple-choice/train_log/log.dataset_swag_"),
        "squad": ("../question-answering/train_log/log.dataset_squad_"),
        "conll2003": ("../token-classification/train_log/log.dataset_conll2003_"),
        "mscoco2017": ("./log/log.dataset_mscoco2017_")
    }
    batch_size_set = [1, 4, 8, 16, 32, 64, 128, 256]
    for dataset_pair, batch_size in itertools.product(dataset_filename.items(), batch_size_set):
        dataset_name, filename_prefix = dataset_pair
        title = dataset_name + f"-batch_size{batch_size}"
        filename = filename_prefix + str(batch_size)
        plot_dataset_distribute(filename, title)

if __name__ == "__main__":
    plot_dataset_distribute_figure()
