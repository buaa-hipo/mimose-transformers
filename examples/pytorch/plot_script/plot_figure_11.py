import matplotlib.pyplot as plt
import json

def get_size_and_mem(f):
  iter_f = iter(f)
  res = {}
  map = {}

  for line in iter_f:
    if "memory_count" in line:
      slice = line[line.find("{"): -1]
      map = json.loads(slice)

  for key, value in map.items():
    for elm in value:
      res[int(key)] = float(elm)/(1024**3)
  return res

def get_size_and_mem_cv(f):
  iter_f = iter(f)
  res={}
  map = {}
  seqlen = 0
  for line in iter_f:
    if "torch.Size" in line:
      slice = line[line.find("[")+1 : line.find("]")]
      nums = slice.split(",")
      seqlen = int(nums[2].strip()) * int(nums[3].strip())

    if "MB" in line:
      memory = float(line[0 : line.find("M")])/(1024)
      map[seqlen] = memory
  return map

f_mc_dc4 = open("dataset/mc_swag_roberta/dc4.log.train_roberta.07010446", 'r', encoding='utf-8')
f_mc_dc5 = open("dataset/mc_swag_roberta/dc5.log.train_roberta.07010523", 'r', encoding='utf-8')
f_mc_dc6 = open("dataset/mc_swag_roberta/dc6.log.train_roberta.07010558", 'r', encoding='utf-8')
f_mc_dc7 = open("dataset/mc_swag_roberta/dc7.log.train_roberta.07010631", 'r', encoding='utf-8')
f_mc_dc8 = open("dataset/mc_swag_roberta/dc8.log.train_roberta.07010705", 'r', encoding='utf-8')

mc_f_list = [
f_mc_dc4,
f_mc_dc5,
f_mc_dc6,
f_mc_dc7,
f_mc_dc8,
]

f_qa_womax_dc4 = open("dataset/qa_bert_womax/log.train_dc4_womax.2022-07-01-07-39-03", 'r', encoding='utf-8')
f_qa_womax_dc5 = open("dataset/qa_bert_womax/log.train_dc5_womax.2022-07-01-08-47-57", 'r', encoding='utf-8')
f_qa_womax_dc6 = open("dataset/qa_bert_womax/log.train_dc6_womax.2022-07-01-09-51-55", 'r', encoding='utf-8')
f_qa_womax_dc7 = open("dataset/qa_bert_womax/log.train_dc7_womax.2022-07-01-10-52-32", 'r', encoding='utf-8')
f_qa_womax_dc8 = open("dataset/qa_bert_womax/log.train_dc8_womax.2022-07-01-11-51-52", 'r', encoding='utf-8')

qa_womax_f_list = [
f_qa_womax_dc4,
f_qa_womax_dc5,
f_qa_womax_dc6,
f_qa_womax_dc7,
f_qa_womax_dc8,
]

f_qa_xlnet_dc5 = open("dataset/qa_xlnet/log.train_dc5_xlnet.2022-07-01-13-21-33", 'r', encoding='utf-8')
f_qa_xlnet_dc6 = open("dataset/qa_xlnet/log.train_dc6_xlnet.2022-07-01-23-06-58", 'r', encoding='utf-8')
f_qa_xlnet_dc7 = open("dataset/qa_xlnet/log.train_dc7_xlnet.2022-07-02-00-46-46", 'r', encoding='utf-8')
f_qa_xlnet_dc8 = open("dataset/qa_xlnet/log.train_dc8_xlnet.2022-07-02-02-24-58", 'r', encoding='utf-8')

qa_xlnet_f_list = [
f_qa_xlnet_dc5,
f_qa_xlnet_dc6,
f_qa_xlnet_dc7,
f_qa_xlnet_dc8,
]

f_tc_dc4 = open("dataset/tc_glue_bert_qqp/log.train_dc4_qqp.2022-07-02-03-59-00", 'r', encoding='utf-8')
f_tc_dc5 = open("dataset/tc_glue_bert_qqp/log.train_dc5_qqp.2022-07-02-04-53-29", 'r', encoding='utf-8')
f_tc_dc6 = open("dataset/tc_glue_bert_qqp/log.train_dc6_qqp.2022-07-02-05-47-02", 'r', encoding='utf-8')
f_tc_dc7 = open("dataset/tc_glue_bert_qqp/log.train_dc7_1_adamw.07042042", 'r', encoding='utf-8')
f_tc_dc8 = open("dataset/tc_glue_bert_qqp/log.train_dc8_1_adamw.07042137", 'r', encoding='utf-8')

tc_f_list = [
f_tc_dc4,
f_tc_dc5,
f_tc_dc6,
f_tc_dc7,
f_tc_dc8,
]

f_res50_dc16 = open("dataset/resnet50/resnet50_dc16.log.2022-10-07-08-28-48", 'r', encoding='utf-8')
f_res101_dc16 = open("dataset/resnet101/resnet101_dc16-8.log.2022-10-06-13-21-13", 'r', encoding='utf-8')

def plot(f_list, filename, s_size=4):
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams['figure.figsize'] = (10, 5)
  plt.tight_layout()
  tick_fontsize = 26
  lable_fontsize = 26
  legend_fontsize = 22
  if filename == "QA_XLNET_memory_consumption":
    i = 5
  else:
    i = 4
  count = 0
  for f in f_list:
    res = get_size_and_mem(f)
    plt.scatter(res.keys(), res.values(), s=s_size, label="MB-"+str(i))
    i += 1
    count += 1
  ymin = 1.6
  ymax = 8.1
  plt.ylim(ymin, ymax)
  plt.xlabel("seqlen", fontsize=lable_fontsize)
  plt.ylabel("Memory Consumption (GB)", fontsize=lable_fontsize)
  plt.xticks(fontsize=tick_fontsize)
  plt.yticks(fontsize=tick_fontsize)
  legend = plt.legend(fontsize = legend_fontsize, loc="upper left", ncol = 1)

  for k in range(count):
    legend.legendHandles[k]._sizes = [300]
  # plt.savefig("../"+filename+".pdf", bbox_inches='tight')
  plt.show()


plot(mc_f_list, "MC_memory_consumption")
plot(qa_womax_f_list, "QA_WOMAX_memory_consumption", 3)
plot(qa_xlnet_f_list, "QA_XLNET_memory_consumption", 3)
plot(tc_f_list, "TC_memory_consumption", 3)

def plot_cv(f, filename, s_size=6, xmin = 400000, xmax = 1500000, ymin = 8, ymax = 16):
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams['figure.figsize'] = (10, 5)
  plt.tight_layout()
  tick_fontsize = 26
  lable_fontsize = 26
  legend_fontsize = 22

  res = get_size_and_mem_cv(f)
  plt.scatter(res.keys(), res.values(), s=s_size, label="MB-16")

  plt.ylim(ymin, ymax)
  plt.xlim(xmin, xmax)
  plt.xlabel("height x width", fontsize=lable_fontsize)
  plt.ylabel("Memory Consumption (GB)", fontsize=lable_fontsize)
  plt.yticks(fontsize=tick_fontsize)
  plt.xticks(fontsize=tick_fontsize)
  ax = plt.gca()
  ax.xaxis.get_offset_text().set_fontsize(tick_fontsize-2)
  legend = plt.legend(fontsize = legend_fontsize, loc="upper left", ncol = 1)
  legend.legendHandles[0]._sizes = [300]
  # plt.savefig("../"+filename+".pdf", bbox_inches='tight')
  plt.show()


plot_cv(f = f_res50_dc16, 
    filename = "RES50_memory_consumption",
    xmin = 390000, xmax = 1500000, ymin = 8, ymax = 15.5
    )
plot_cv(f = f_res101_dc16, 
    filename = "RES101_memory_consumption",
    xmin = 390000, xmax = 1500000, ymin = 6.5, ymax = 14.1
    )
