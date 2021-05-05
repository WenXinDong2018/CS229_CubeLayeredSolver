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


###--------------------------------------------Search Experiments -------------------------------------------###

###Layer 1
#1. A* search without options using saved checkpoint of the layer 1 baseline model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_baseline/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_baseline/ --language python --nnet_batch_size 10000 --start_idx 900

#2. A* search without options using saved checkpoint of the layer 1 dynamic difficulty uniform distribution model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_dynamic_difficulty_25/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_dynamic_difficulty_25/ --language python --nnet_batch_size 10000 --start_idx 900

#3. A* search without options using saved checkpoint of the layer 1 dynamic difficulty fixed length model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer1_dynamic_difficulty_25_fixed_length/current/ --env cube3_layer1 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer1_dynamic_difficulty_25_fixed_length/ --language python --nnet_batch_size 10000 --start_idx 900

###Layer 2
#1. First Layer Fixed: A* search without options using saved checkpoint of the layer 2 model. solve 100 cubes
python search_methods/astar.py --states data/cube3_layer2/test/data_0.pkl --model saved_models/cube3layer2_dynamic_difficulty_25_fixed/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/fixed_layer_no_options --language python --nnet_batch_size 10000 --start_idx 900
##With Options
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3layer2_random_data_gen/current/ --env cube3_layer2 --weight 0.6 --batch_size 1000 --results_dir results/cube3layer2_dynamic_difficulty_25_fixed/fixed_layer_and_options --language python --nnet_batch_size 10000 --start_idx 900 --options --option_name layer2

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
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full/ --language python --nnet_batch_size 10000 --start_idx 900
#2. Save as above, no options, test on first two layers fixed
python search_methods/astar.py --states data/cube3_layer3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_layer3/ --language python --nnet_batch_size 10000 --start_idx 900
#3. A* search with options using saved checkpoint of the cube3 model. solve 100 cubes
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 1000 --results_dir results/cube3_full_options/ --language python --nnet_batch_size 10000 --start_idx 900 --options
