#!/bin/bash

var=$((4))

for i in `seq 2 4`
do
        ssh -f 10.10.$i.1 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.$i.1' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank $var --tensor-size $2"
        var=$((var+1))

        ssh -f 10.10.$i.2 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.$i.2' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank $var --tensor-size $2"
        var=$((var+1))

        ssh -f 10.10.$i.3 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.$i.3' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank $var --tensor-size $2"
        var=$((var+1))

        ssh -f 10.10.$i.4 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.$i.4' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank $var --tensor-size $2"
        var=$((var+1))
done

ssh -f 10.10.1.2 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.1.2' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank 1 --tensor-size $2"
ssh -f 10.10.1.3 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.1.3' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank 2 --tensor-size $2"
ssh -f 10.10.1.4 "export GLOO_SOCKET_IFNAME=\$(ifconfig | grep -B 1 '10.10.1.4' | head -n 1 | cut -d: -f1) ; nohup python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank 3 --tensor-size $2"

export GLOO_SOCKET_IFNAME=$(ifconfig | grep -B 1 '10.10.1.1' | head -n 1 | cut -d: -f1)
python3 majority_msr.py --master-ip 10.10.1.1 --num-nodes $1 --rank 0 --tensor-size $2