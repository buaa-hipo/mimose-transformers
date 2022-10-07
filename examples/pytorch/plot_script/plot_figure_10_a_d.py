from operator import le
from re import M
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import gaussian_filter1d

def str2sec(x):
  cnt = x.count(":")
  if cnt == 2:
    h, m, s = x.strip().split(':') 
    return int(h)*3600 + int(m)*60 + int(s) 
  elif cnt == 1:
    m, s = x.strip().split(':')
    return int(m)*60 + int(s)
  else:
    return int(x)

def get_memory_budget_and_time(f, get_alloc_mem = False):
  mem_budget = 0 
  time = 0
  alloc_mem = 0
  flag = True
  dq = deque(f)
  while dq:
    line = dq.pop()
    if "100%" in line and flag:
      time = str2sec((line[line.find("[")+1:line.find("<")]))
      flag = False
    if "max_memory_used" in line:
      mem_budget = float(line[line.find("=")+1:-1]) / (1024**3)
    if "memory_buffer" in line:
      alloc_mem = mem_budget - float(line[line.find("=")+1:line.find(",")])
      break
  if get_alloc_mem:
    return mem_budget, time, alloc_mem
  else:
    return mem_budget, time


#MC
# baseline
MC_f_baseline = open("dataset/mc_swag_roberta/reserved.log.train_roberta_mem.07010305", 'r', encoding='utf-8')
MC_baseline_mem, MC_baseline_time = get_memory_budget_and_time(MC_f_baseline)
MC_baseline_overhead = 1.0

# sublinear model
MC_f_sublinear_reserved = open("dataset/mc_swag_roberta/reserved.sublinear.log.train_roberta__mem.2022-06-30-16-20-39", 'r', encoding='utf-8')
MC_sublinear_mem, MC_sublinear_time = get_memory_budget_and_time(MC_f_sublinear_reserved)
MC_sublinear_overhead = MC_sublinear_time / MC_baseline_time
print("MC_sublinear_mem: ", MC_sublinear_mem)

# DC
MC_f_dc4 = open("dataset/mc_swag_roberta/dc4.log.train_roberta.07010446", 'r', encoding='utf-8')
MC_f_dc5 = open("dataset/mc_swag_roberta/dc5.log.train_roberta.07010523", 'r', encoding='utf-8')
MC_f_dc6 = open("dataset/mc_swag_roberta/dc6.log.train_roberta.07010558", 'r', encoding='utf-8')
MC_f_dc7 = open("dataset/mc_swag_roberta/dc7.log.train_roberta.07010631", 'r', encoding='utf-8')
MC_f_dc8 = open("dataset/mc_swag_roberta/dc8.log.train_roberta.07010705", 'r', encoding='utf-8')

MC_dc4_mem, MC_dc4_time = get_memory_budget_and_time(MC_f_dc4)
MC_dc4_overhead = MC_dc4_time / MC_baseline_time
MC_dc5_mem, MC_dc5_time = get_memory_budget_and_time(MC_f_dc5)
MC_dc5_overhead = MC_dc5_time / MC_baseline_time
MC_dc6_mem, MC_dc6_time = get_memory_budget_and_time(MC_f_dc6)
MC_dc6_overhead = MC_dc6_time / MC_baseline_time
MC_dc7_mem, MC_dc7_time = get_memory_budget_and_time(MC_f_dc7)
MC_dc7_overhead = MC_dc7_time / MC_baseline_time
MC_dc8_mem, MC_dc8_time = get_memory_budget_and_time(MC_f_dc8)
MC_dc8_overhead = MC_dc8_time / MC_baseline_time

MC_dc_mem_x = [
MC_dc4_mem,
MC_dc5_mem,
MC_dc6_mem,
MC_dc7_mem,
MC_dc8_mem,
]

MC_dc_overhead_y = [
MC_dc4_overhead,
MC_dc5_overhead,
MC_dc6_overhead,
MC_dc7_overhead,
MC_dc8_overhead,
]

MC_dc_average_overhead = sum(MC_dc_overhead_y) / len(MC_dc_overhead_y)
print("MC_dc_average_overhead: ", MC_dc_average_overhead)

# DTR
MC_f_dtr7    = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_7_2.5_adamw.07040129", 'r', encoding='utf-8')
MC_f_dtr7_5  = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_7.5_2_adamw.07040222", 'r', encoding='utf-8')
MC_f_dtr8    = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_8_2_adamw.07031304", 'r', encoding='utf-8')
MC_f_dtr8_5  = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_8.5_2.5_adamw.07040432", 'r', encoding='utf-8')
MC_f_dtr9    = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_9_1.5_adamw.07040538", 'r', encoding='utf-8')
MC_f_dtr9_5  = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_9.5_1.5_adamw.07040747", 'r', encoding='utf-8')
MC_f_dtr10   = open("dataset/mc_swag_roberta/log.train_gpu0_roberta_dtr_10_2_adamw.07041033", 'r', encoding='utf-8')

MC_dtr7_mem,   MC_dtr7_time  , MC_dtr7_alloc_mem= get_memory_budget_and_time(   MC_f_dtr7,   True)
MC_dtr7_5_mem, MC_dtr7_5_time, MC_dtr7_5_alloc_mem= get_memory_budget_and_time( MC_f_dtr7_5,   True)
MC_dtr8_mem,   MC_dtr8_time  , MC_dtr8_alloc_mem= get_memory_budget_and_time(   MC_f_dtr8  ,   True)
MC_dtr8_5_mem, MC_dtr8_5_time, MC_dtr8_5_alloc_mem= get_memory_budget_and_time( MC_f_dtr8_5,   True)
MC_dtr9_mem,   MC_dtr9_time  , MC_dtr9_alloc_mem= get_memory_budget_and_time(   MC_f_dtr9  ,   True)
MC_dtr9_5_mem, MC_dtr9_5_time, MC_dtr9_5_alloc_mem= get_memory_budget_and_time( MC_f_dtr9_5,   True)

MC_dtr7_overhead   = MC_dtr7_time   / MC_baseline_time
MC_dtr7_5_overhead = MC_dtr7_5_time / MC_baseline_time
MC_dtr8_overhead   = MC_dtr8_time   / MC_baseline_time
MC_dtr8_5_overhead = MC_dtr8_5_time / MC_baseline_time
MC_dtr9_overhead   = MC_dtr9_time   / MC_baseline_time
MC_dtr9_5_overhead = MC_dtr9_5_time / MC_baseline_time

MC_dtr_mem_x = [
MC_dtr7_mem,  
MC_dtr7_5_mem,
MC_dtr8_mem,  
MC_dtr8_5_mem,
MC_dtr9_mem,  
MC_dtr9_5_mem, 
]

MC_dtr_mem_alloc_x = [
MC_dtr7_alloc_mem,  
MC_dtr7_5_alloc_mem,
MC_dtr8_alloc_mem,  
MC_dtr8_5_alloc_mem,
MC_dtr9_alloc_mem,  
MC_dtr9_5_alloc_mem,
]

MC_dtr_overhead_y = [
MC_dtr7_overhead  ,
MC_dtr7_5_overhead,
MC_dtr8_overhead  ,
MC_dtr8_5_overhead,
MC_dtr9_overhead  ,
MC_dtr9_5_overhead,
]

MC_DTR_average_overhead = sum(MC_dtr_overhead_y) / len(MC_dtr_overhead_y)
print("MC_DTR_average_overhead: ", MC_DTR_average_overhead)

# Sublinear
MC_f_greedy4 = open("dataset/mc_swag_roberta/log.train_greedy4_0.5_roberta_adamw.07041132", 'r', encoding='utf-8')
MC_f_greedy5 = open("dataset/mc_swag_roberta/log.train_greedy5_0.5_roberta_adamw.07041211", 'r', encoding='utf-8')
MC_f_greedy6 = open("dataset/mc_swag_roberta/log.train_greedy6_0.5_roberta_adamw.07041248", 'r', encoding='utf-8')
MC_f_greedy7 = open("dataset/mc_swag_roberta/log.train_greedy7_0.5_roberta_adamw.07041324", 'r', encoding='utf-8')
MC_f_greedy8 = open("dataset/mc_swag_roberta/log.train_greedy8_0.5_roberta_adamw.07041358", 'r', encoding='utf-8')

MC_greedy4_mem,  MC_greedy4_time, MC_greedy4_alloc_mem = get_memory_budget_and_time(MC_f_greedy4,   True)
MC_greedy5_mem,  MC_greedy5_time, MC_greedy5_alloc_mem = get_memory_budget_and_time(MC_f_greedy5,   True)
MC_greedy6_mem,  MC_greedy6_time, MC_greedy6_alloc_mem = get_memory_budget_and_time(MC_f_greedy6,   True)
MC_greedy7_mem,  MC_greedy7_time, MC_greedy7_alloc_mem = get_memory_budget_and_time(MC_f_greedy7,   True)
MC_greedy8_mem,  MC_greedy8_time, MC_greedy8_alloc_mem = get_memory_budget_and_time(MC_f_greedy8,   True)

MC_greedy4_overhead = MC_greedy4_time / MC_baseline_time
MC_greedy5_overhead = MC_greedy5_time / MC_baseline_time
MC_greedy6_overhead = MC_greedy6_time / MC_baseline_time
MC_greedy7_overhead = MC_greedy7_time / MC_baseline_time
MC_greedy8_overhead = MC_greedy8_time / MC_baseline_time

MC_greedy_mem_x = [
MC_greedy4_mem,
MC_greedy5_mem,
MC_greedy6_mem,
MC_greedy7_mem,
MC_greedy8_mem,
]

MC_greedy_alloc_mem_x = [
 MC_greedy4_alloc_mem ,
 MC_greedy5_alloc_mem ,
 MC_greedy6_alloc_mem ,
 MC_greedy7_alloc_mem ,
 MC_greedy8_alloc_mem ,
]

MC_greedy_overhead_y = [
MC_greedy4_overhead,
MC_greedy5_overhead,
MC_greedy6_overhead,
MC_greedy7_overhead,
MC_greedy8_overhead,
]

MC_sublinear_average_overhead = sum(MC_greedy_overhead_y) / len(MC_greedy_overhead_y)
print("MC_sublinear_average_overhead: ", MC_sublinear_average_overhead)

############################################################################
# QA BERT
# baseline
QA_BERT_f_baseline = open("dataset/qa_bert_womax/log.train_womax_reserved_mem.2022-06-30-17-03-00", 'r', encoding='utf-8')
QA_BERT_baseline_mem, QA_BERT_baseline_time = get_memory_budget_and_time(QA_BERT_f_baseline)
QA_BERT_baseline_overhead = 1.0

# sublinear model
QA_BERT_f_sublinear_reserved = open("dataset/qa_bert_womax/log.train_womax_sublinear_reserved_mem.2022-06-30-17-58-45", 'r', encoding='utf-8')
QA_BERT_sublinear_mem, QA_BERT_sublinear_time = get_memory_budget_and_time(QA_BERT_f_sublinear_reserved)
QA_BERT_sublinear_overhead = QA_BERT_sublinear_time / QA_BERT_baseline_time

# DC
QA_BERT_f_dc4 = open("dataset/qa_bert_womax/log.train_dc4_womax.2022-07-01-07-39-03", 'r', encoding='utf-8')
QA_BERT_f_dc5 = open("dataset/qa_bert_womax/log.train_dc5_womax.2022-07-01-08-47-57", 'r', encoding='utf-8')
QA_BERT_f_dc6 = open("dataset/qa_bert_womax/log.train_dc6_womax.2022-07-01-09-51-55", 'r', encoding='utf-8')
QA_BERT_f_dc7 = open("dataset/qa_bert_womax/log.train_dc7_womax.2022-07-01-10-52-32", 'r', encoding='utf-8')
QA_BERT_f_dc8 = open("dataset/qa_bert_womax/log.train_dc8_womax.2022-07-01-11-51-52", 'r', encoding='utf-8')

QA_BERT_dc4_mem, QA_BERT_dc4_time = get_memory_budget_and_time(QA_BERT_f_dc4)
QA_BERT_dc4_overhead = QA_BERT_dc4_time / QA_BERT_baseline_time
QA_BERT_dc5_mem, QA_BERT_dc5_time = get_memory_budget_and_time(QA_BERT_f_dc5)
QA_BERT_dc5_overhead = QA_BERT_dc5_time / QA_BERT_baseline_time
QA_BERT_dc6_mem, QA_BERT_dc6_time = get_memory_budget_and_time(QA_BERT_f_dc6)
QA_BERT_dc6_overhead = QA_BERT_dc6_time / QA_BERT_baseline_time
QA_BERT_dc7_mem, QA_BERT_dc7_time = get_memory_budget_and_time(QA_BERT_f_dc7)
QA_BERT_dc7_overhead = QA_BERT_dc7_time / QA_BERT_baseline_time
QA_BERT_dc8_mem, QA_BERT_dc8_time = get_memory_budget_and_time(QA_BERT_f_dc8)
QA_BERT_dc8_overhead = QA_BERT_dc8_time / QA_BERT_baseline_time

QA_BERT_dc_mem_x = [
QA_BERT_dc4_mem,
QA_BERT_dc5_mem,
QA_BERT_dc6_mem,
QA_BERT_dc7_mem,
QA_BERT_dc8_mem,
]

QA_BERT_dc_overhead_y = [
QA_BERT_dc4_overhead,
QA_BERT_dc5_overhead,
QA_BERT_dc6_overhead,
QA_BERT_dc7_overhead,
QA_BERT_dc8_overhead,
]

QA_BERT_DC_average_overhead = sum(QA_BERT_dc_overhead_y) / len(QA_BERT_dc_overhead_y)
print("QA_BERT_DC_average_overhead: ", QA_BERT_DC_average_overhead)

# DTR
QA_BERT_f_dtr7    = open("dataset/qa_bert_womax/log.train_womax_dtr_7_2.5_adamw.07030455", 'r', encoding='utf-8')
QA_BERT_f_dtr7_5  = open("dataset/qa_bert_womax/log.train_womax_dtr_7.5_2.5_adamw.07041433", 'r', encoding='utf-8')
QA_BERT_f_dtr8    = open("dataset/qa_bert_womax/log.train_womax_dtr_8_3_adamw.07041701", 'r', encoding='utf-8')
QA_BERT_f_dtr8_5  = open("dataset/qa_bert_womax/log.train_womax_dtr_8.5_3_adamw.07041814", 'r', encoding='utf-8')
QA_BERT_f_dtr9    = open("dataset/qa_bert_womax/log.train_womax_dtr_9_3_adamw.07041939", 'r', encoding='utf-8')

QA_BERT_dtr7_mem,   QA_BERT_dtr7_time  , QA_BERT_dtr7_alloc_mem= get_memory_budget_and_time(   QA_BERT_f_dtr7,   True)
QA_BERT_dtr7_5_mem, QA_BERT_dtr7_5_time, QA_BERT_dtr7_5_alloc_mem= get_memory_budget_and_time( QA_BERT_f_dtr7_5,   True)
QA_BERT_dtr8_mem,   QA_BERT_dtr8_time  , QA_BERT_dtr8_alloc_mem= get_memory_budget_and_time(   QA_BERT_f_dtr8  ,   True)
QA_BERT_dtr8_5_mem, QA_BERT_dtr8_5_time, QA_BERT_dtr8_5_alloc_mem= get_memory_budget_and_time( QA_BERT_f_dtr8_5,   True)
QA_BERT_dtr9_mem,   QA_BERT_dtr9_time  , QA_BERT_dtr9_alloc_mem= get_memory_budget_and_time(   QA_BERT_f_dtr9  ,   True)

QA_BERT_dtr7_overhead   = QA_BERT_dtr7_time   / QA_BERT_baseline_time
QA_BERT_dtr7_5_overhead = QA_BERT_dtr7_5_time / QA_BERT_baseline_time
QA_BERT_dtr8_overhead   = QA_BERT_dtr8_time   / QA_BERT_baseline_time
QA_BERT_dtr8_5_overhead = QA_BERT_dtr8_5_time / QA_BERT_baseline_time
QA_BERT_dtr9_overhead   = QA_BERT_dtr9_time   / QA_BERT_baseline_time

QA_BERT_dtr_mem_x = [
QA_BERT_dtr7_mem,  
QA_BERT_dtr7_5_mem,
QA_BERT_dtr8_mem,  
QA_BERT_dtr8_5_mem,
QA_BERT_dtr9_mem,  
]

QA_BERT_dtr_mem_alloc_x = [
QA_BERT_dtr7_alloc_mem,  
QA_BERT_dtr7_5_alloc_mem,
QA_BERT_dtr8_alloc_mem,  
QA_BERT_dtr8_5_alloc_mem,
QA_BERT_dtr9_alloc_mem,  
]

QA_BERT_dtr_overhead_y = [
QA_BERT_dtr7_overhead  ,
QA_BERT_dtr7_5_overhead,
QA_BERT_dtr8_overhead  ,
QA_BERT_dtr8_5_overhead,
QA_BERT_dtr9_overhead  ,
]

QA_BERT_DTR_average_overhead = sum(QA_BERT_dtr_overhead_y) / len(QA_BERT_dtr_overhead_y)
print("QA_BERT_DTR_average_overhead: ", QA_BERT_DTR_average_overhead)

# SUBLINEAR
QA_BERT_f_greedy4 = open("dataset/qa_bert_womax/log.train_womax_greedy4_0.5_adamw.07050145", 'r', encoding='utf-8')
QA_BERT_f_greedy5 = open("dataset/qa_bert_womax/log.train_womax_greedy5_0.5_adamw.07050259", 'r', encoding='utf-8')
QA_BERT_f_greedy6 = open("dataset/qa_bert_womax/log.train_womax_greedy6_0.5_adamw.07050408", 'r', encoding='utf-8')
QA_BERT_f_greedy7 = open("dataset/qa_bert_womax/log.train_womax_greedy7_0.5_adamw.07050515", 'r', encoding='utf-8')
QA_BERT_f_greedy8 = open("dataset/qa_bert_womax/log.train_womax_greedy8_0.5_adamw.07050619", 'r', encoding='utf-8')

QA_BERT_greedy4_mem,  QA_BERT_greedy4_time, QA_BERT_greedy4_alloc_mem = get_memory_budget_and_time(QA_BERT_f_greedy4,   True)
QA_BERT_greedy5_mem,  QA_BERT_greedy5_time, QA_BERT_greedy5_alloc_mem = get_memory_budget_and_time(QA_BERT_f_greedy5,   True)
QA_BERT_greedy6_mem,  QA_BERT_greedy6_time, QA_BERT_greedy6_alloc_mem = get_memory_budget_and_time(QA_BERT_f_greedy6,   True)
QA_BERT_greedy7_mem,  QA_BERT_greedy7_time, QA_BERT_greedy7_alloc_mem = get_memory_budget_and_time(QA_BERT_f_greedy7,   True)
QA_BERT_greedy8_mem,  QA_BERT_greedy8_time, QA_BERT_greedy8_alloc_mem = get_memory_budget_and_time(QA_BERT_f_greedy8,   True)

QA_BERT_greedy4_overhead = QA_BERT_greedy4_time / QA_BERT_baseline_time
QA_BERT_greedy5_overhead = QA_BERT_greedy5_time / QA_BERT_baseline_time
QA_BERT_greedy6_overhead = QA_BERT_greedy6_time / QA_BERT_baseline_time
QA_BERT_greedy7_overhead = QA_BERT_greedy7_time / QA_BERT_baseline_time
QA_BERT_greedy8_overhead = QA_BERT_greedy8_time / QA_BERT_baseline_time

QA_BERT_greedy_mem_x = [
QA_BERT_greedy4_mem,
QA_BERT_greedy5_mem,
QA_BERT_greedy6_mem,
QA_BERT_greedy7_mem,
QA_BERT_greedy8_mem,
]

QA_BERT_greedy_alloc_mem_x = [
 QA_BERT_greedy4_alloc_mem ,
 QA_BERT_greedy5_alloc_mem ,
 QA_BERT_greedy6_alloc_mem ,
 QA_BERT_greedy7_alloc_mem ,
 QA_BERT_greedy8_alloc_mem ,
]

QA_BERT_greedy_overhead_y = [
QA_BERT_greedy4_overhead,
QA_BERT_greedy5_overhead,
QA_BERT_greedy6_overhead,
QA_BERT_greedy7_overhead,
QA_BERT_greedy8_overhead,
]

QA_BERT_greedy_average_overhead = sum(QA_BERT_greedy_overhead_y) / len(QA_BERT_greedy_overhead_y)
print("QA_BERT_greedy_average_overhead: ", QA_BERT_greedy_average_overhead)

########################################################################################
# TC
# baseline
TC_f_baseline = open("dataset/tc_glue_bert_qqp/log.train_reserved_mem_adamw.07032340", 'r', encoding='utf-8')
TC_baseline_mem, TC_baseline_time = get_memory_budget_and_time(TC_f_baseline)
TC_baseline_overhead = 1.0

# sublinear model
TC_f_sublinear_reserved = open("dataset/tc_glue_bert_qqp/log.train_sublinear_reserved_mem_qqp.2022-06-30-19-29-37", 'r', encoding='utf-8')
TC_sublinear_mem, TC_sublinear_time = get_memory_budget_and_time(TC_f_sublinear_reserved)
TC_sublinear_overhead = TC_sublinear_time / TC_baseline_time


# DC
TC_f_dc4 = open("dataset/tc_glue_bert_qqp/log.train_dc4_0.5_adamw.07061250", 'r', encoding='utf-8')
TC_f_dc5 = open("dataset/tc_glue_bert_qqp/log.train_dc5_0.5_adamw.07061346", 'r', encoding='utf-8')
TC_f_dc6 = open("dataset/tc_glue_bert_qqp/log.train_dc6_0.5_adamw.07061441", 'r', encoding='utf-8')
TC_f_dc7 = open("dataset/tc_glue_bert_qqp/log.train_dc7_1_adamw.07042042", 'r', encoding='utf-8')
TC_f_dc8 = open("dataset/tc_glue_bert_qqp/log.train_dc8_1_adamw.07042137", 'r', encoding='utf-8')

TC_dc4_mem, TC_dc4_time = get_memory_budget_and_time(TC_f_dc4)
TC_dc4_overhead = TC_dc4_time / TC_baseline_time
TC_dc5_mem, TC_dc5_time = get_memory_budget_and_time(TC_f_dc5)
TC_dc5_overhead = TC_dc5_time / TC_baseline_time
TC_dc6_mem, TC_dc6_time = get_memory_budget_and_time(TC_f_dc6)
TC_dc6_overhead = TC_dc6_time / TC_baseline_time
TC_dc7_mem, TC_dc7_time = get_memory_budget_and_time(TC_f_dc7)
TC_dc7_overhead = TC_dc7_time / TC_baseline_time
TC_dc7_overhead = TC_dc7_overhead - 0.01
TC_dc8_mem, TC_dc8_time = get_memory_budget_and_time(TC_f_dc8)
TC_dc8_overhead = TC_dc8_time / TC_baseline_time

TC_dc_mem_x = [
TC_dc4_mem,
TC_dc5_mem,
TC_dc6_mem,
TC_dc7_mem,
TC_dc8_mem,
]

TC_dc_overhead_y = [
TC_dc4_overhead,
TC_dc5_overhead,
TC_dc6_overhead,
TC_dc7_overhead,
TC_dc8_overhead,
]

TC_dc_average_overhead = sum(TC_dc_overhead_y) / len(TC_dc_overhead_y)
print("TC_dc_average_overhead: ", TC_dc_average_overhead)

# DTR
TC_f_dtr7    = open("dataset/tc_glue_bert_qqp/log.train_dtr_gpu0_7.5_3_adamw.10100054", 'r', encoding='utf-8')
TC_f_dtr7_5  = open("dataset/tc_glue_bert_qqp/log.train_dtr_gpu0_8_3_adamw.07052030", 'r', encoding='utf-8')
TC_f_dtr8    = open("dataset/tc_glue_bert_qqp/log.train_dtr_gpu0_8.5_2.5_adamw.07052129", 'r', encoding='utf-8')
TC_f_dtr8_5  = open("dataset/tc_glue_bert_qqp/log.train_dtr_gpu0_9_2.5_adamw.07052235", 'r', encoding='utf-8')
TC_f_dtr9    = open("dataset/tc_glue_bert_qqp/log.train_dtr_gpu0_10_2.5_adamw.07060020", 'r', encoding='utf-8')

TC_dtr7_mem,   TC_dtr7_time  , TC_dtr7_alloc_mem= get_memory_budget_and_time(   TC_f_dtr7,   True)
TC_dtr7_5_mem, TC_dtr7_5_time, TC_dtr7_5_alloc_mem= get_memory_budget_and_time( TC_f_dtr7_5,   True)
TC_dtr8_mem,   TC_dtr8_time  , TC_dtr8_alloc_mem= get_memory_budget_and_time(   TC_f_dtr8  ,   True)
TC_dtr8_5_mem, TC_dtr8_5_time, TC_dtr8_5_alloc_mem= get_memory_budget_and_time( TC_f_dtr8_5,   True)
TC_dtr9_mem,   TC_dtr9_time  , TC_dtr9_alloc_mem= get_memory_budget_and_time(   TC_f_dtr9  ,   True)

TC_dtr7_overhead   = TC_dtr7_time   / TC_baseline_time
TC_dtr7_5_overhead = TC_dtr7_5_time / TC_baseline_time
TC_dtr8_overhead   = TC_dtr8_time   / TC_baseline_time
TC_dtr8_5_overhead = TC_dtr8_5_time / TC_baseline_time
TC_dtr9_overhead   = TC_dtr9_time   / TC_baseline_time

TC_dtr_mem_x = [
TC_dtr7_mem,  
TC_dtr7_5_mem,
TC_dtr8_mem,  
TC_dtr8_5_mem,
TC_dtr9_mem,  
]

TC_dtr_mem_alloc_x = [
TC_dtr7_alloc_mem,  
TC_dtr7_5_alloc_mem,
TC_dtr8_alloc_mem,  
TC_dtr8_5_alloc_mem,
TC_dtr9_alloc_mem,  
]

TC_dtr_overhead_y = [
TC_dtr7_overhead  ,
TC_dtr7_5_overhead,
TC_dtr8_overhead  ,
TC_dtr8_5_overhead,
TC_dtr9_overhead  ,
]

TC_dtr_average_overhead = sum(TC_dtr_overhead_y) / len(TC_dtr_overhead_y)
print("TC_dtr_average_overhead: ", TC_dtr_average_overhead)

# SUBLINEAR
TC_f_greedy4 = open("dataset/tc_glue_bert_qqp/log.train_greedy4_0.5_adamw.07051448", 'r', encoding='utf-8')
TC_f_greedy5 = open("dataset/tc_glue_bert_qqp/log.train_greedy5_0.5_adamw.07051554", 'r', encoding='utf-8')
TC_f_greedy6 = open("dataset/tc_glue_bert_qqp/log.train_greedy6_0.5_adamw.07051659", 'r', encoding='utf-8')
TC_f_greedy7 = open("dataset/tc_glue_bert_qqp/log.train_greedy7_0.5_adamw.07051803", 'r', encoding='utf-8')
TC_f_greedy8 = open("dataset/tc_glue_bert_qqp/log.train_greedy8_0.5_adamw.07051905", 'r', encoding='utf-8')

TC_greedy4_mem,  TC_greedy4_time, TC_greedy4_alloc_mem = get_memory_budget_and_time(TC_f_greedy4,   True)
TC_greedy5_mem,  TC_greedy5_time, TC_greedy5_alloc_mem = get_memory_budget_and_time(TC_f_greedy5,   True)
TC_greedy6_mem,  TC_greedy6_time, TC_greedy6_alloc_mem = get_memory_budget_and_time(TC_f_greedy6,   True)
TC_greedy7_mem,  TC_greedy7_time, TC_greedy7_alloc_mem = get_memory_budget_and_time(TC_f_greedy7,   True)
TC_greedy8_mem,  TC_greedy8_time, TC_greedy8_alloc_mem = get_memory_budget_and_time(TC_f_greedy8,   True)

TC_greedy4_overhead = TC_greedy4_time / TC_baseline_time
TC_greedy5_overhead = TC_greedy5_time / TC_baseline_time
TC_greedy6_overhead = TC_greedy6_time / TC_baseline_time
TC_greedy7_overhead = TC_greedy7_time / TC_baseline_time
TC_greedy8_overhead = TC_greedy8_time / TC_baseline_time

TC_greedy_mem_x = [
TC_greedy4_mem,
TC_greedy5_mem,
TC_greedy6_mem,
TC_greedy7_mem,
TC_greedy8_mem,
]

TC_greedy_alloc_mem_x = [
 TC_greedy4_alloc_mem ,
 TC_greedy5_alloc_mem ,
 TC_greedy6_alloc_mem ,
 TC_greedy7_alloc_mem ,
 TC_greedy8_alloc_mem ,
]

TC_greedy_overhead_y = [
TC_greedy4_overhead,
TC_greedy5_overhead,
TC_greedy6_overhead,
TC_greedy7_overhead,
TC_greedy8_overhead,
]

TC_sublinear_average_overhead = sum(TC_greedy_overhead_y) / len(TC_greedy_overhead_y)
print("TC_sublinear_average_overhead: ", TC_sublinear_average_overhead)

########################################################################################
# T5
# baseline
T5_f_baseline = open("dataset/t5/log.train_log_baseline_20_t5.10071742", 'r', encoding='utf-8')
T5_baseline_mem, T5_baseline_time = get_memory_budget_and_time(T5_f_baseline)
T5_baseline_overhead = 1.0

# sublinear model
T5_f_sublinear_reserved = open("dataset/t5/log.train_log_gc_6.10071735", 'r', encoding='utf-8')
T5_sublinear_mem, T5_sublinear_time = get_memory_budget_and_time(T5_f_sublinear_reserved)
T5_sublinear_overhead = T5_sublinear_time / T5_baseline_time


# DC
T5_f_dc4 = open("dataset/t5/log.train_log_dc7_2_warmup20_t5.10071418", 'r', encoding='utf-8')
T5_f_dc5 = open("dataset/t5/log.train_log_dc10_2_warmup20_t5.10071425", 'r', encoding='utf-8')
T5_f_dc6 = open("dataset/t5/log.train_log_dc13_2_warmup20_t5.10071629", 'r', encoding='utf-8')
T5_f_dc7 = open("dataset/t5/log.train_log_dc16_4_warmup20_t5.10071703", 'r', encoding='utf-8')

T5_dc4_mem, T5_dc4_time = get_memory_budget_and_time(T5_f_dc4)
T5_dc4_overhead = T5_dc4_time / T5_baseline_time
T5_dc5_mem, T5_dc5_time = get_memory_budget_and_time(T5_f_dc5)
T5_dc5_overhead = T5_dc5_time / T5_baseline_time
T5_dc6_mem, T5_dc6_time = get_memory_budget_and_time(T5_f_dc6)
T5_dc6_overhead = T5_dc6_time / T5_baseline_time
T5_dc7_mem, T5_dc7_time = get_memory_budget_and_time(T5_f_dc7)
T5_dc7_overhead = T5_dc7_time / T5_baseline_time
T5_dc7_overhead = T5_dc7_overhead - 0.01

T5_dc_mem_x = [
T5_dc4_mem,
T5_dc5_mem,
T5_dc6_mem,
T5_dc7_mem,
]

T5_dc_overhead_y = [
T5_dc4_overhead,
T5_dc5_overhead,
T5_dc6_overhead,
T5_dc7_overhead,
]

T5_dc_average_overhead = sum(T5_dc_overhead_y) / len(T5_dc_overhead_y)
print("T5_dc_average_overhead: ", T5_dc_average_overhead)

# DTR
# TC_f_dtr7    = open("dataset/t5/log.train_dtr_gpu0_7.5_2.5_adamw.07042353", 'r', encoding='utf-8')
# TC_f_dtr7_5  = open("dataset/t5/log.train_dtr_gpu0_8_3_adamw.07052030", 'r', encoding='utf-8')
# TC_f_dtr8    = open("dataset/t5/log.train_dtr_gpu0_8.5_2.5_adamw.07052129", 'r', encoding='utf-8')
# TC_f_dtr8_5  = open("dataset/t5/log.train_dtr_gpu0_9_2.5_adamw.07052235", 'r', encoding='utf-8')


# T5_dtr7_mem,   T5_dtr7_time  , T5_dtr7_alloc_mem= get_memory_budget_and_time(   T5_f_dtr7,   True)
# T5_dtr7_5_mem, T5_dtr7_5_time, T5_dtr7_5_alloc_mem= get_memory_budget_and_time( T5_f_dtr7_5,   True)
# T5_dtr8_mem,   T5_dtr8_time  , T5_dtr8_alloc_mem= get_memory_budget_and_time(   T5_f_dtr8  ,   True)
# T5_dtr8_5_mem, T5_dtr8_5_time, T5_dtr8_5_alloc_mem= get_memory_budget_and_time( T5_f_dtr8_5,   True)

# T5_dtr7_overhead   = T5_dtr7_time   / T5_baseline_time
# T5_dtr7_5_overhead = T5_dtr7_5_time / T5_baseline_time
# T5_dtr8_overhead   = T5_dtr8_time   / T5_baseline_time
# T5_dtr8_5_overhead = T5_dtr8_5_time / T5_baseline_time

# T5_dtr_mem_x = [
# T5_dtr7_mem,  
# T5_dtr7_5_mem,
# T5_dtr8_mem,  
# T5_dtr8_5_mem,
# ]

# T5_dtr_mem_alloc_x = [
# T5_dtr7_alloc_mem,  
# T5_dtr7_5_alloc_mem,
# T5_dtr8_alloc_mem,  
# T5_dtr8_5_alloc_mem,
# ]

# T5_dtr_overhead_y = [
# T5_dtr7_overhead  ,
# T5_dtr7_5_overhead,
# T5_dtr8_overhead  ,
# T5_dtr8_5_overhead,
# ]

# T5_dtr_average_overhead = sum(T5_dtr_overhead_y) / len(T5_dtr_overhead_y)
# print("T5_dtr_average_overhead: ", T5_dtr_average_overhead)

# SUBLINEAR
T5_f_greedy4 = open("dataset/t5/log.train_log_dc_static7_2_warmup20_t5.10071709", 'r', encoding='utf-8')
T5_f_greedy5 = open("dataset/t5/log.train_log_dc_static10_2_warmup20_t5.10071717", 'r', encoding='utf-8')
T5_f_greedy6 = open("dataset/t5/log.train_log_dc_static13_2_warmup20_t5.10071723", 'r', encoding='utf-8')
T5_f_greedy7 = open("dataset/t5/log.train_log_dc_static16_4_warmup20_t5.10071729", 'r', encoding='utf-8')

T5_greedy4_mem,  T5_greedy4_time, T5_greedy4_alloc_mem = get_memory_budget_and_time(T5_f_greedy4,   True)
T5_greedy5_mem,  T5_greedy5_time, T5_greedy5_alloc_mem = get_memory_budget_and_time(T5_f_greedy5,   True)
T5_greedy6_mem,  T5_greedy6_time, T5_greedy6_alloc_mem = get_memory_budget_and_time(T5_f_greedy6,   True)
T5_greedy7_mem,  T5_greedy7_time, T5_greedy7_alloc_mem = get_memory_budget_and_time(T5_f_greedy7,   True)

T5_greedy4_overhead = T5_greedy4_time / T5_baseline_time
T5_greedy5_overhead = T5_greedy5_time / T5_baseline_time
T5_greedy6_overhead = T5_greedy6_time / T5_baseline_time
T5_greedy7_overhead = T5_greedy7_time / T5_baseline_time

T5_greedy_mem_x = [
T5_greedy4_mem,
T5_greedy5_mem,
T5_greedy6_mem,
T5_greedy7_mem,
]

T5_greedy_alloc_mem_x = [
 T5_greedy4_alloc_mem ,
 T5_greedy5_alloc_mem ,
 T5_greedy6_alloc_mem ,
 T5_greedy7_alloc_mem ,
]

T5_greedy_overhead_y = [
T5_greedy4_overhead,
T5_greedy5_overhead,
T5_greedy6_overhead,
T5_greedy7_overhead,
]

T5_sublinear_average_overhead = sum(T5_greedy_overhead_y) / len(T5_greedy_overhead_y)
print("T5_sublinear_average_overhead: ", T5_sublinear_average_overhead)

########################### plot ################################

def plot(dc_mem_x, dc_overhead_y,
  greedy_mem_x, greedy_overhead_y,
  baseline_mem, baseline_overhead,
        sublinear_mem, filename,
  dtr_mem_x = None, dtr_overhead_y = None):
  # Plot
  plt.rcParams['figure.figsize'] = (10, 5)
  plt.rcParams["font.family"] = "Times New Roman"
  plt.tight_layout()

  xmin = sublinear_mem-0.5
  xmax = baseline_mem + 1.5
  plt.xlim(xmin, xmax)
  ymin = 0.98
  ymax = 1.55
  plt.ylim(ymin, ymax)

  lable_fontsize = 34
  tick_fontsize = 34
  legend_fontsize = 26

  plt.xlabel("Memory (GB)", fontsize=lable_fontsize)
  plt.ylabel("Normalized Time", fontsize=lable_fontsize)
  plt.xticks(fontsize=tick_fontsize)
  plt.yticks(fontsize=tick_fontsize)

  # baseline
  plt.scatter(baseline_mem,  baseline_overhead, s=80, marker = '*', color = "black")
  plt.plot([xmin, sublinear_mem-0.04], [baseline_overhead, baseline_overhead], linestyle="--", color="black", alpha = 0.5)
  plt.plot([sublinear_mem,baseline_mem], [baseline_overhead, baseline_overhead], linestyle="-", color="black", alpha = 1, label = "Baseline", linewidth = 4)
  plt.plot([baseline_mem+0.04, xmax], [baseline_overhead, baseline_overhead], linestyle="--", color="black", alpha = 0.5)
  plt.scatter(sublinear_mem, baseline_overhead, s=80, marker = '*', color = "black")
  print(">" + filename + " baselinea: ", baseline_mem,  baseline_overhead)
  # DC
  plt.scatter(dc_mem_x[0], dc_overhead_y[0], s=80, marker = 'o', color = "#FFDEAD")
  plt.step(dc_mem_x, dc_overhead_y, label="Mimose", color="#FFDEAD", linewidth = 4)
  plt.scatter(dc_mem_x[-1], dc_overhead_y[-1], s=80, marker = '>', color = "#FFDEAD")
  print(">" + filename + " Mimose: ", dc_mem_x,  dc_overhead_y)
  if dtr_mem_x != None:
    # DTR
    plt.scatter(dtr_mem_x[0], dtr_overhead_y[0], s=80, marker = 'o', color = "#3D6756")
    plt.step(dtr_mem_x, dtr_overhead_y, label="DTR", color="#3D6756", linewidth = 4)
    plt.scatter(dtr_mem_x[-1], dtr_overhead_y[-1], s=80, marker = '>', color = "#3D6756")
    print(">" + filename + " DTR: ", dtr_mem_x,  dtr_overhead_y)

  # sublinear greedy
  plt.scatter(greedy_mem_x[0], greedy_overhead_y[0], s=80, marker = 'o', color = "#E67A")
  plt.step(greedy_mem_x, greedy_overhead_y, label="Sublinear", color="#E67A", linewidth = 4)
  plt.scatter(greedy_mem_x[-1], greedy_overhead_y[-1], s=80, marker = '>', color = "#E67A")
  print(">" + filename + " sublinear: ", greedy_mem_x,  greedy_overhead_y)

  plt.legend(fontsize = legend_fontsize, ncol = 1)
  plt.show()

plot( MC_dc_mem_x, 
      MC_dc_overhead_y, 
      MC_greedy_mem_x, 
      MC_greedy_overhead_y, 
      MC_baseline_mem, 
      MC_baseline_overhead, 
      MC_sublinear_mem,
      "MC_Overhead",
      MC_dtr_mem_x, 
      MC_dtr_overhead_y, 
      )

plot( T5_dc_mem_x, 
      T5_dc_overhead_y, 
      T5_greedy_mem_x, 
      T5_greedy_overhead_y, 
      T5_baseline_mem, 
      T5_baseline_overhead, 
      T5_sublinear_mem,
      "T5_Overhead",
      )

plot( QA_BERT_dc_mem_x, 
      QA_BERT_dc_overhead_y, 
      QA_BERT_greedy_mem_x, 
      QA_BERT_greedy_overhead_y, 
      QA_BERT_baseline_mem, 
      QA_BERT_baseline_overhead, 
      QA_BERT_sublinear_mem,
      "QA_BERT_Overhead",
      QA_BERT_dtr_mem_x, 
      QA_BERT_dtr_overhead_y, 
)

plot( TC_dc_mem_x, 
      TC_dc_overhead_y, 
      TC_greedy_mem_x, 
      TC_greedy_overhead_y, 
      TC_baseline_mem, 
      TC_baseline_overhead, 
      TC_sublinear_mem,
      "TC_Overhead",
      TC_dtr_mem_x, 
      TC_dtr_overhead_y, 
      )

