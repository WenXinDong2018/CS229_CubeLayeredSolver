#training layer 1 with the default settings

python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_baseline --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30

#states_per_update: How many states to train on before checking if target network should be updated"