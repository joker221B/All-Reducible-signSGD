import argparse
import torch
import logging
import time
import math
from statistics import mean
from torch import distributed as dist

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo", \
            init_method="tcp://" + master_ip + ":6585", \
            rank=rank, \
            world_size=world_size)

def calc_majority(t_local):
    return torch.sum(t_local.int(), dim=0)

def scatter_async(t, t_local):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk_size = t.size()[0]/world_size

    start_inds = [int(((rank + i) % world_size) * chunk_size) for i in range(1, world_size)]
    end_inds = [start_inds[i-1] +  chunk_size for i in range(1, world_size)]
    t_local[0] = t[int(rank * chunk_size) : int(((rank + 1) * chunk_size))]
    send_reqs = []
    recv_reqs = []

    for i in range(1, world_size):
        send_reqs.append(dist.isend(t[int(start_inds[i-1]) : int(end_inds[i-1])], (rank + i) % world_size))
    for i in range(1, world_size):
        recv_reqs.append(dist.irecv(t_local[i], src=(rank - i + world_size) % world_size))
    for i in range(1, world_size):
        send_reqs[i-1].wait()
        recv_reqs[i-1].wait()

def gather_async(t):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk_size = t.size()[0]/world_size

    start_inds = [int(((rank + i) % world_size) * chunk_size) for i in range(1, world_size)]
    end_inds = [start_inds[i-1] +  chunk_size for i in range(1, world_size)]
    send_reqs = []
    recv_reqs = []

    for i in range(1, world_size):
        send_reqs.append(dist.isend(t[int(rank*chunk_size) : int((rank+1)*chunk_size)], (rank + i) % world_size))
    for i in range(1, world_size):
        recv_reqs.append(dist.irecv(t[int(start_inds[i-1]) : int(end_inds[i-1])], (rank + i) % world_size))
    for i in range(1, world_size):
        send_reqs[i-1].wait()
        recv_reqs[i-1].wait()

def all_gather_recursive_hd(t, left, right, tensor_start_ind, tensor_cur_size):
    if left == right:
        return
    size = right - left + 1
    mid = math.floor((left + right)/2)
    rank = dist.get_rank()
    partner = rank + math.floor(size/2) if rank <= mid else rank - math.floor(size/2)

    if rank <= mid:
        all_gather_recursive_hd(t, left, mid, tensor_start_ind, tensor_cur_size/2)
        dist.send(t[int(tensor_start_ind) : int(tensor_start_ind + tensor_cur_size/2)], dst=partner)
        tmp = torch.empty(int(tensor_cur_size/2), dtype=torch.bool)
        dist.recv(tmp, src=partner)
        t[int(tensor_start_ind + tensor_cur_size/2) : int(tensor_start_ind + tensor_cur_size)] = tmp
        return

    all_gather_recursive_hd(t, mid + 1, right, tensor_start_ind + tensor_cur_size/2, tensor_cur_size/2)
    tmp = torch.empty(int(tensor_cur_size/2), dtype=torch.bool)
    dist.recv(tmp, src=partner)
    t[int(tensor_start_ind) : int(tensor_start_ind + tensor_cur_size/2)] = tmp
    dist.send(t[int(tensor_start_ind + tensor_cur_size/2) : int(tensor_start_ind + tensor_cur_size)], dst=partner)
    return

def main(tensor_size):
    # t = torch.ones(tensor_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t = (torch.randn(tensor_size) < 0.25)
    chunk_size = t.size()[0]/world_size
    t_local = torch.empty((world_size, int(tensor_size/world_size)), dtype=torch.bool)
    s = time.time()
    scatter_async(t, t_local)

    t_majority = calc_majority(t_local)
    t_majority_bool = t_majority > world_size/2
    t[int(rank * chunk_size) : int(((rank + 1) * chunk_size))] = t_majority

    gather_async(t)
    e = time.time()
    print(e-s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--tensor-size", "-t", required=True, type=int)

    args = parser.parse_args()
    init_process(master_ip=args.master_ip, rank=args.rank, world_size=args.num_nodes)

    main(args.tensor_size)
