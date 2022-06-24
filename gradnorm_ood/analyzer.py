import sys
import os
import json
import tqdm
import matplotlib.pyplot as plt
import numpy as np

def analyzer(json_path, mode="linear"):
    log = []
    with open(json_path, mode="r") as f:
        log_txt = f.readlines()
        for l in log_txt:
            if len(l) < 2:
                continue
            log.append(json.loads(l))
    ignore = ["data_time", "eta_seconds", "iteration", "time"]
    log_key_dict = dict()
    iter_list = []
    assert len(log), "Get empty json log."
    for k in log[0].keys():
        if k in ignore:
            continue
        if type(log[0][k]) in [int, float]:
            log_key_dict[k] = []
    print("Collecting metrics...")
    for idx in tqdm.trange(len(log)):
        iter_list.append(int(log[idx]["iteration"]))
        for k in log_key_dict.keys():
            log_key_dict[k].append(float(log[idx][k]))
    for k in log_key_dict.keys():
        fig, ax = plt.subplots()
        if mode == "linear":
            ax.plot(iter_list, log_key_dict[k])
            ax.set(xlabel='iter', ylabel="loss", title=k)
        elif mode == "log":
            ax.plot(iter_list, 10 * np.log10(np.array(log_key_dict[k])))
            ax.set(xlabel='iter', ylabel="loss(dB)", title=k)
        fig.savefig(k + ".png")
        del fig, ax

if __name__ == "__main__":
    analyzer("metrics.json", mode="linear")
    # analyzer(sys.argv[0])