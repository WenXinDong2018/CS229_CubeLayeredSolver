###-------------------------------------------- Final Report -------------------------------------------###
###-------------------------------------------- Training Experiments -------------------------------------------###

### Layer 1
#1. training layer 1 with the default settings [DONE]
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3layer1_baseline --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30
#2. training layer 1 with dynamic difficulty, fixed length [DONE]
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3layer1_dynamic_difficulty_25_fixed --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
### Layer 2
#1. training layer 2 with the default settings [DONE]
python ctg_approx/avi.py --env cube3_layer2 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3layer2_baseline --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30
#2. training layer 2 with dynamic difficulty, fixed length [TODO]
python ctg_approx/avi.py --env cube3_layer2 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3layer2_dynamic_difficulty_25_fixed --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
### Layer 3
#1. training layer 3 with the default settings [TODO]
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3layer3_baseline --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30
### Multihead Model Baseline [TODO]
python ctg_approx/avi_multihead.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3multihead_baseline --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30



###--------------------------------------------Search Experiments (100 cubes, no options)-------------------------------------------###
#1. A* search, layer 1 baseline model.[DONE]
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_baseline/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_baseline/ --language python --nnet_batch_size 10000 --start_idx 900
#2. A* search, layer 1 dynamic curriculum model.[TODO]
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/final_cube3layer1_dynamic_difficulty_25_fixed/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/final_cube3layer1_dynamic_difficulty_25_fixed/ --language python --nnet_batch_size 10000 --start_idx 900
#3. A* search, layer 2 baseline model.[TODO]
python search_methods/astar.py --states data/cube3_layer2/test/data_0.pkl --model saved_models/final_cube3layer2_baseline/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/final_cube3layer2_baseline --language python --nnet_batch_size 10000 --start_idx 900
#4. A* search, layer 2 dynamic curriculum model.[TODO]
python search_methods/astar.py --states data/cube3_layer2/test/data_0.pkl --model saved_models/final_cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/final_cube3layer2_dynamic_difficulty_25_fixed/ --language python --nnet_batch_size 10000 --start_idx 900

##Sequential Model [TODO]

##Multihead Model Baseline [TODO]


###-------------------------------------------- Not Final Report  -------------------------------------------###
###-------------------------------------------- Training Experiments -------------------------------------------###
###Layer 1
#1. training layer 1 with the default settings
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_baseline --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30
#Note: states_per_update: How many states to train on before checking if target network should be updated"
#Note: nnet_name is the unique id of each experiment

#2. training layer 1 with dynamic difficulty, uniform distribution
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_dynamic_difficulty_25 --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25

#3. training layer 1 with dynamic difficulty, fixing length of training examples
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer1_dynamic_difficulty_25_fixed --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty

###Layer 2
#1.training layer 2 with with dynamic difficulty, fixing length of training examples, use loss_threshold of 0.2
python ctg_approx/avi.py --env cube3_layer2 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer2_dynamic_difficulty_25_fixed --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty

###Layer 3
#1.training layer 3 with with dynamic difficulty, fixing length of training examples
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
#2.Same as above but use loss_threshold of 0.06
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed_lt --max_itrs 1000000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
#3.Same as above but use loss_threshold of 0.3
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed_lt_03 --max_itrs 1000000 --loss_thresh 0.3 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
#4.Same as above but use loss_threshold of 0.5
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed_lt_05 --max_itrs 1000000 --loss_thresh 0.5 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
#5.Same as above but use loss_threshold of 0.2, states_per_update = 5000000
python ctg_approx/avi.py --env cube3 --states_per_update 5000000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed_spux10 --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty
#6.training layer 3 with without dynamic curriculum
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_baseline --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30
#7.training layer 3 with with dynamic curriculum, uniform distribution
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_uniform --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25
#8.training layer 3 with with dynamic curriculum, fixed length, only update dynamix_back when target model updates to current model
python ctg_approx/avi.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3layer3_dynamic_difficulty_25_fixed_target_full --max_itrs 46200 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --dynamic_back_max --dynamic_back_max_per 25 --fixed_difficulty


###Multi-head
#1.train
python ctg_approx/avi_multihead.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name cube3multihead_baseline --max_itrs 1000000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30

###--------------------------------------------Search Experiments -------------------------------------------###

###Layer 1
#1. A* search without options using saved checkpoint of the layer 1 baseline model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_baseline/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_baseline/ --language python --nnet_batch_size 10000 --start_idx 900

#2. A* search without options using saved checkpoint of the layer 1 dynamic difficulty uniform distribution model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_dynamic_difficulty_25/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_dynamic_difficulty_25/ --language python --nnet_batch_size 10000 --start_idx 900

#3. A* search without options using saved checkpoint of the layer 1 dynamic difficulty fixed length model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_dynamic_difficulty_25_fixed_length/ --language python --nnet_batch_size 10000 --start_idx 900

#4. A* search without options using saved checkpoint of the layer 1 dynamic difficulty fixed length model. solve 1000 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_dynamic_difficulty_25_fixed_length/ --language python --nnet_batch_size 10000 --start_idx 0

###Layer 2
#1. First Layer Fixed: A* search without options using saved checkpoint of the layer 2 model. solve 100 cubes
python search_methods/astar.py --states data/cube3_layer2/test/data_0.pkl --model saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/fixed_layer_no_options --language python --nnet_batch_size 10000 --start_idx 900
##With Options
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/fixed_layer_and_options --language python --nnet_batch_size 10000 --start_idx 900 --options --option_name layer2

#2. No fixed layers:  A* search without options using saved checkpoint of the layer 2 model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/no_fixed_layer_no_options --language python --nnet_batch_size 10000 --start_idx 900
## With Options
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/no_fixed_layer_and_options --language python --nnet_batch_size 10000 --start_idx 900 --options --option_name layer2

###Layer 3
#1. First two layers fixed: A* search without options using saved checkpoint of the layer 3 model. solve 100 cubes
python search_methods/astar.py --states data/cube3_layer3/test/data_0.pkl --model saved_models/cube3layer3_dynamic_difficulty_25_fixed_lt_03/current/ --env cube3_layer3 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer3_dynamic_difficulty_25_fixed_lt_03/fixed_layer_no_options --language python --nnet_batch_size 10000 --start_idx 900
#With Options
python search_methods/astar.py --states data/cube3_layer3/test/data_0.pkl --model saved_models/cube3layer3_dynamic_difficulty_25_fixed_lt_03/current/ --env cube3_layer3 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer3_dynamic_difficulty_25_fixed_lt_03/fixed_layer_and_options --language python --nnet_batch_size 10000 --start_idx 900 --options --option_name layer3

#２. No fixed layers:: A* search without options using saved checkpoint of the layer 3 model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer3_dynamic_difficulty_25_fixed_lt_03/current/ --env cube3_layer3 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer3_dynamic_difficulty_25_fixed_lt_03/no_fixed_layer_no_options --language python --nnet_batch_size 10000 --start_idx 900
#With Options
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer3_dynamic_difficulty_25_fixed_lt_03/current/ --env cube3_layer3 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer3_dynamic_difficulty_25_fixed_lt_03/no_fixed_layer_and_options --language python --nnet_batch_size 10000 --start_idx 900 --options --option_name layer3


###DeepCubeA
#1. A* search without options using saved checkpoint of the cube3 model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_500/ --language python --nnet_batch_size 10000 --start_idx 500
#2. Save as above, no options, test on first two layers fixed
python search_methods/astar.py --states data/cube3_layer3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_layer3/ --language python --nnet_batch_size 10000 --start_idx 900
#3. A* search with options using saved checkpoint of the cube3 model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_options/ --language python --nnet_batch_size 10000 --start_idx 900 --options
#4. A* search without options using saved checkpoint of the cube3 model. solve 100 cubes. Goal is to solve layer 1
python search_methods/astar.py --states data/cube3/test/data_0.pkl  --model saved_models/cube3/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_layer1/ --language python --nnet_batch_size 10000 --start_idx 900
#5. A* search without options using saved checkpoint of the cube3 model. solve 100 cubes. Goal is to solve layer 1 and 2
python search_methods/astar.py --states data/cube3/test/data_0.pkl  --model saved_models/cube3/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_layer1_and_2/ --language python --nnet_batch_size 10000 --start_idx 900
#6. A* search without options using saved checkpoint of the cube3 model. solve 100 cubes. Goal is to solve 2 given layer 1
python search_methods/astar.py --states data/cube3_layer2/test/data_0.pkl  --model saved_models/cube3/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_layer2/ --language python --nnet_batch_size 10000 --start_idx 900


###Sequential model
#1.Solve 100 cubes
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential/
#2.Solve 500 cubes
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 500 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential_500/
#3.Solve 1000 cubes
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 0 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential_1000/
#3. Solves 100 cubes. Use our layer 3 instead of deepcubea as layer 3
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3layer3_dynamic_difficulty_25_fixed_lt_03/current/ --results_dir results/cube3_sequential_our_layer3/
#4. sequential model with roll_out option on data_1 (data_1 has 2000 data)
python search_methods/sequential.py --states data/cube3/test/data_1.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 1900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential_option_rollout/ --options --option_name roll_out
#5. sequential model without option on data_1
python search_methods/sequential.py --states data/cube3/test/data_1.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 1900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential_data1/
#6. sequential model with roll_out option on data_0
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential_option_rollout_data0/ --options --option_name roll_out


# multihead model
python search_methods/sequential_multi_head.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir saved_models/cube3multihead_baseline/current --results_dir results/cube3_multihead_baseline/

# sequential model with roll_out option on data_1
python search_methods/sequential.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir_layer1 saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --model_dir_layer2 saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --model_dir_layer3 saved_models/cube3/current/ --results_dir results/cube3_sequential/

