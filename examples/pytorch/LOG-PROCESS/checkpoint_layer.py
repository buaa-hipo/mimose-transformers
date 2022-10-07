import os
from log_process import read_log, plot_memory_line
import matplotlib.pyplot as plt



def read_max_memory(filename):
    _, memory = read_log(filename)
    for key, value in memory.items():
        memory[key] = [max(value)]
    return memory

def get_last_file(file_list):
    last_data = 0
    last_file = None
    for file in file_list:
        data = int(file.split('.')[-1])
        if data > last_data:
            last_data = data
            last_file = file
    return last_file

def find_filepath(file_dir, search_file):
    files = os.listdir(file_dir)
    file_path = {}
    for target in search_file:
        match_file_list = []
        for filename in files:
            if target in filename:
                match_file_list.append(filename)
        assert match_file_list, f"can't find {target}"
        file_path[target] = os.path.join(file_dir, get_last_file(match_file_list))
    return file_path


if __name__ == "__main__":
    search_file = ["checkpoint_layer_0_0"]
    for i in [1, 2, 4, 6, 8]:
        search_file.append(f"checkpoint_layer_{i}_0")
        # search_file.append(f"checkpoint_layer_0_{i}")
    search_file.append("checkpoint_layer_0_2")
    file_path = find_filepath("./log/checkpoint_layer/", search_file)
    
    # file_path.pop("checkpoint_layer_0_4")
    
    fig = plt.figure()
    for file, path in file_path.items():
        memory = read_max_memory(path)
        plot_memory_line(memory, file)
    plt.xlabel("seq length")
    plt.ylabel("memory / MB")
    plt.legend()
    plt.savefig("fig/checkpoint_layer/diff_layer_number.png")
    plt.close(fig)
    