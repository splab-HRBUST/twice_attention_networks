# author: muzhan
import os
import sys
import time
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # print("\ngpu_status0 = ",gpu_status[2])
    # print("gpu_status1 = ",gpu_status[6])
    # print("gpu_status2 = ",gpu_status[10])
    # print("gpu_status3 = ",gpu_status[14])
    # print("gpu_status4 = ",gpu_status[18])
    # print("gpu_status5 = ",gpu_status[22])
    # print("gpu_status6 = ",gpu_status[26])
    # print("gpu_status7 = ",gpu_status[30])
    gpu_memorys = []
    for i in range(8):
        gpu_memory = int(gpu_status[2+4*i].split('/')[0].split('M')[0].strip())
        gpu_memorys.append(gpu_memory)
    # gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    # print("gpu_memorys = ",gpu_memorys)
    return gpu_memorys
def get_idx(list_):
    str1 = []
    for i in list_:
        if i<5000:
            print("i = ",i," --",str(list_.index(i)))
            str1.append(str(list_.index(i)))
            list_[list_.index(i)] = -1
    print("str1 = ",str1)
    return str1


def narrow_setup(interval=2):
    i = 0
    # list.index(min(list))
    while len(get_idx(gpu_info()))<1:  # 设置最少使用gpu个数
        gpu_memorys = gpu_info()
        gpu_memory = min(gpu_memorys)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    
    device_ids_list = get_idx(gpu_info())
    print("device_ids_list = ",device_ids_list)
    max_num = 2 #设置gpu个数
    if len(device_ids_list)>max_num-1:
        device_ids_list = device_ids_list[0:]
    device_num = len(device_ids_list)
    print("device_num = ",device_num)
    device_ids = ",".join(device_ids_list)
    print("device_ids = ",str(device_ids))
    # cmd = "python -m torch.distributed.launch  --nproc_per_node="+str(device_num)+" model_main2.py --num_epochs=200 --track=logical --features=spect --device_ids="+str(device_ids)+"> test.txt"
    eval_device=str(device_ids[-1])
    cmd = "python -m torch.distributed.launch  --nproc_per_node="+str(1)+" model_main2.py  --eval  --eval_output=Unet_scores_innovation23_dev_cqt_MGD.txt  --features=spect --track=logical --model_path=models/model_logical_spect_200_8_0.001_Unet_innovation23/epoch_30.pth --device_ids="+str(eval_device)+" > tes_eval.txt"
    del eval_device
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()