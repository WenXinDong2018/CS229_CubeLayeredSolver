#training layer 1 with the default settings

python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_baseline --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30

#Note: states_per_update: How many states to train on before checking if target network should be updated"
#Note: nnet_name is the unique id of each experiment

#training layer 1 with dynamic curriculum

python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_dynamic_difficulty_25 --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25


#training layer 1 with dynamic curriculum, fixing difficulty of training examples in each lesson

python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_dynamic_difficulty_25_fixed --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty

#training layer 2 with random state training examples in each lesson

python ctg_approx/avi.py --env cube3_layer2 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer2_random_data_gen --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --uniform_data_gen
