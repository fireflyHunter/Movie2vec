import os
import sys
import numpy as np
import pandas as pd
from utils import *

from multiprocessing import Pool
import queue
import threading
import time

file_path = "./data"


def load_movie_sents(file):
    items, ratings = [], []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i % 2 == 0:
                items.append(np.array(line.strip().split(",")).astype(np.int32))
            else:
                ratings.append(np.array(line.strip().split(",")).astype(np.float32))
    return np.array(items), np.array(ratings)

"""
load items and ratings
"""
ml_10m_movie_items, ml_10m_movie_ratings = load_movie_sents(
    os.path.join(file_path, "10m-movie_sents_low_count_removed.txt"))
print(len(ml_10m_movie_items), len(ml_10m_movie_ratings))

"""
load training item list
"""
ml_10m_movie_items_vector_index = np.load('./data/vectors/ml-10m-vectorsindex_map').tolist()

reverse = np.zeros(np.max(ml_10m_movie_items_vector_index) + 1)
for i, _id in enumerate(ml_10m_movie_items_vector_index):
    reverse[_id] = i

"""
replace item by item index
slow
"""
ml_10m_items_to_train = []
sp = ShowProcess(len(ml_10m_movie_items))
for u in ml_10m_movie_items:
    temp = [reverse[i] for i in u if i in ml_10m_movie_items_vector_index]
    temp = [reverse[i] for i in u]
    ml_10m_items_to_train.append(temp)
    sp.show_process()
sp.close()

"""
multi process: replace item by item index
"""
# result = queue.Queue()
# pool = Pool(processes=42)
# # pool = Pool(processes=1)
# ml_10m_items_to_train = []


# def replace(u):
#     temp = [ml_10m_movie_items_vector_index.index(i) for i in u if i in ml_10m_movie_items_vector_index]
#     return temp


# def pool_th():
#     # for u in ml_10m_movie_items[:1]:
#     for u in ml_10m_movie_items:
#         try:
#             result.put(pool.apply_async(replace, args=(u, ml_10m_movie_items_vector_index)))
#         except:
#             break
#
#
# def result_th():
#     j = 0
#
#     sp = ShowProcess(len(ml_10m_movie_items))
#     while 1:
#         a = result.get().get()  # 获取子进程返回值
#
#         ml_10m_items_to_train.append(a)
#         j += 1
#
#         sp.show_process()
#
#         # if j == len(ml_10m_movie_items[:1]):
#         if j == len(ml_10m_movie_items):
#             pool.terminate()  # 结束所有子进程
#             break
#     sp.close()
#
#
# t1 = threading.Thread(target=pool_th)
# t2 = threading.Thread(target=result_th)
# t1.start()
# t2.start()
# t1.join()
# t2.join()
# pool.close()
#
# """
# save
# """
# np.save("./data/10m-items.npy", ml_10m_items_to_train)
# np.save("./data/10m-ratings.npy", ml_10m_movie_ratings)

