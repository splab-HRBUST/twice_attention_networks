# 将每一批次的得分和标签，拼接到一起。
import numpy as np
def cat_scores(dev_scores,dev_y,length):

    dev_target_scores = np.array([])
    dev_nontarget_scores = np.array([])

    for j in range(length):
        if(dev_y[j]==1):
            dev_target_scores=np.append(dev_target_scores,dev_scores[j])
        else:
            dev_nontarget_scores=np.append(dev_nontarget_scores,dev_scores[j])
    return dev_target_scores,dev_nontarget_scores