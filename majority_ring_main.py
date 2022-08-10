import argparse
import torch
import logging
import time
import math
from statistics import mean
from torch import distributed as dist
import numpy as np

offsets = torch.Tensor([48, 40, 36, 34, 33]).int()

binary_offsets = torch.Tensor([[16], [8], [4], [2], [1]]).float()

def convert_to_bool(t, bits_len=5):
    t_ = torch.reshape(t, (t.size()[0], 1))
    t_ = torch.flatten(torch.bitwise_and(t_, offsets[5 - bits_len : ]))
    return t_.bool()

def convert_to_int(t, bits_len=5):
    t_ = torch.reshape(t, (-1, bits_len))
    return torch.flatten(torch.matmul(t_.float(), binary_offsets[5 - bits_len : ]).int())

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo", \
            init_method="tcp://" + master_ip + ":6585", \
            rank=rank, \
            world_size=world_size)

def ring_all_reduce(tensor, tensor_bool, t_bool_reduced):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # neighborhood
    left = ((rank - 1) + world_size) % world_size
    right = (rank + 1) % world_size

    #splitting tensor to chunks
    length = len(tensor_bool)
    chunk_size = int(length / world_size)
    residual = length % world_size
    chunk_sizes = [chunk_size] * world_size
    for i in range(residual):
        chunk_sizes[i] += 1

    #chunk_ends: ith chunks's ending index
    chunk_ends = [0] * world_size
    chunk_ends[0] = chunk_sizes[0]
    for i in range(1, world_size):
        chunk_ends[i] = chunk_sizes[i] + chunk_ends[i-1]

    recv_buff = torch.empty(tensor_bool.size(), dtype=torch.bool)

    #reduce
    for i in range(world_size - 1):
        send_index = (rank - i + world_size) % world_size
        recv_index = (rank - i - 1 + world_size) % world_size
        send_start = chunk_ends[send_index] - chunk_sizes[send_index]
        recv_start = chunk_ends[recv_index] - chunk_sizes[recv_index]

        bits_len = int(math.ceil(math.log(i+2, 2)))
        send_buff = convert_to_bool(tensor[int(send_start/5) : int(chunk_ends[send_index]/5)], bits_len)
        send_req = dist.isend(send_buff, right)
        recv_req = dist.irecv(recv_buff[0 : send_buff.size()[0]], src=left)

        recv_req.wait()
        tensor[int(recv_start/5) :  int(chunk_ends[recv_index]/5)].add_(convert_to_int(recv_buff[0 : send_buff.size()[0]], bits_len))
        send_req.wait()

    recv_index = (rank - (world_size - 2) - 1 + world_size) % world_size
    recv_start = chunk_ends[recv_index] - chunk_sizes[recv_index]
    t_bool_reduced[int(recv_start/5) :  int(chunk_ends[recv_index]/5)] = tensor[int(recv_start/5) :  int(chunk_ends[recv_index]/5)] >= world_size/2

    #only send
    for i in range(world_size - 1):
        send_index = (rank - i + 1 + world_size) % world_size
        recv_index = (rank - i + world_size) % world_size
        send_start = chunk_ends[send_index] - chunk_sizes[send_index]
        recv_start = chunk_ends[recv_index] - chunk_sizes[recv_index]

        send_req = dist.isend(t_bool_reduced[int(send_start/5):int(chunk_ends[send_index]/5)], right)
        recv_req = dist.irecv(t_bool_reduced[int(recv_start/5):int(chunk_ends[recv_index]/5)], src=left)
        recv_req.wait()
        send_req.wait()

    del recv_buff

    return

def main(tensor_size):
    t = (torch.randn(tensor_size) < 0.25).int()
    tensor_bool = convert_to_bool(t)
    t_bool_reduced = torch.empty(t.size(), dtype=torch.bool)
    s = time.time()

    ring_all_reduce(t, tensor_bool, t_bool_reduced)
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