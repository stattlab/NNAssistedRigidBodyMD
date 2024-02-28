python3 selector_net_training.py -i x_train_for_bool.pt labels.pt --stop 201 --lr 0.001 --batch_size 1024 --width 70 --depth 5 --gpu 1 --minutes 750
python3 -u energy_net_training.py -i x_train_int.pt yf_train_int_mm01.pt --stop 201 --lr 0.0005 --batch_size 1024 --width 110 --depth 15 --gpu 1 --minutes 1000 --col 3 --torfor 0
