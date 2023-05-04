import tpot2
import sklearn.metrics
import sklearn
from dask.distributed import Client
from dask.distributed import LocalCluster
import argparse
import tpot
import utils
import tpot2
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

def params_LogisticRegression(trial, name=None):
    params = {}
    params['solver'] = trial.suggest_categorical(name=f'solver_{name}',
                                                 choices=[f'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    params['dual'] = False
    params['penalty'] = 'l2'
    params['C'] = trial.suggest_float(f'C_{name}', 1e-4, 1e4, log=True)
    params['l1_ratio'] = None
    if params['solver'] == 'liblinear':
        params['penalty'] = trial.suggest_categorical(name=f'penalty_{name}', choices=['l1', 'l2'])
        if params['penalty'] == 'l2':
            params['dual'] = trial.suggest_categorical(name=f'dual_{name}', choices=[True, False])
        else:
            params['penalty'] = 'l1'

    params['class_weight'] = trial.suggest_categorical(name=f'class_weight_{name}', choices=['balanced'])
    param_grid = {'solver': params['solver'],
                  'penalty': params['penalty'],
                  'dual': params['dual'],
                  'multi_class': 'auto',
                  'l1_ratio': params['l1_ratio'],
                  'C': params['C'],
                  'n_jobs': 1,
                  }
    return param_grid

def main():
    # Read in arguements
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')

    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    experiments = [
                        {
                        'automl': tpot.TPOTClassifier,
                        'exp_name' : 'tpot_untimed_30_gen_roc_auc',
                        'params': {
                                    'scoring': 'roc_auc',
                                    'population_size' : 48, 
                                    'generations' : 30, 
                                    'n_jobs':n_jobs,
                                    'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                    'verbosity': 2, 
                                    'max_time_mins': None,
                                    'max_eval_time_mins' : 5,         
                        },
                        },

                                                                                                                        {
                        'automl': tpot2.TPOTClassifier,
                        'exp_name' : 'tpot2_untimed_30_gen_roc_auc',
                        'params': {
                    
                        'scorers' :     ['roc_auc'],     
                                        'population_size' : 48, 
                                        'generations' : 30, 
                                        'n_jobs':n_jobs,
                                        'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                        'verbose':5, 
                                        'max_time_seconds': None,
                                        'max_eval_time_seconds':60*5, 

                                        'crossover_probability':.10,
                                        'mutate_probability':.90,
                                        'mutate_then_crossover_probability':0,
                                        'crossover_then_mutate_probability':0,

                                        'other_objective_functions' : [tpot2.estimator_objective_functions.number_of_nodes_objective],
                                        'other_objective_functions_weights':[-1],
                                        
                                        'memory_limit':'20GB', 
                                        'preprocessing':False,
                            },
                            },


                        #                                                                                                                         {
                        # 'automl': tpot2.TPOTClassifier,
                        # 'exp_name' : 'tpot2_untimed_30_gen_ensemble_roc_auc',
                        # 'params': {
                    
                        # 'scorers' :     ['roc_auc'],     
                        #                 'population_size' : 48, 
                        #                 'generations' : 30, 
                        #                 'n_jobs':n_jobs,
                        #                 'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                        #                 'verbose':5, 
                        #                 'max_time_seconds': None,
                        #                 'max_eval_time_seconds':60*5, 


                        #                 'root_config_dict' : 'classifiers',
                        #                 'inner_config_dict' : ["selectors", "transformers",'classifiers'],


                        #                 'crossover_probability':.10,
                        #                 'mutate_probability':.90,
                        #                 'mutate_then_crossover_probability':0,
                        #                 'crossover_then_mutate_probability':0,

                        #                 'other_objective_functions' : [tpot2.estimator_objective_functions.number_of_nodes_objective],
                        #                 'other_objective_functions_weights':[-1],
                                        
                        #                 'memory_limit':'20GB', 
                        #                 'preprocessing':False,
                        #     },
                        #     },


                        #                                                                                                                             {
                        # 'automl': tpot2.TPOTClassifier,
                        # 'exp_name' : 'tpot2_untimed_60_gen_roc_auc_SH',
                        # 'params': {
                    
                        # 'scorers' :     ['roc_auc'],     
                        #                 'population_size' : 48, 
                        #                 'generations' :60, 

                        #                 'initial_population_size':48*2,
                        #                 'population_scaling' : .5,
                        #                 'generations_until_end_population' : 50,
                        #                 'budget_range' : [.1,1],
                        #                 'generations_until_end_budget':50,
                        #                 'stepwise_steps':4,

                        #                 'root_config_dict' : 'classifiers',
                        #                 'inner_config_dict' : ["selectors", "transformers",'classifiers'],

                                        # 'crossover_probability':.10,
                                        # 'mutate_probability':.90,
                                        # 'mutate_then_crossover_probability':0,
                                        # 'crossover_then_mutate_probability':0,

                        #                 'n_jobs':n_jobs,
                        #                 'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                        #                 'verbose':5, 
                        #                 'max_time_seconds': None,
                        #                 'max_eval_time_seconds':60*5, 

                        #                 'other_objective_functions' : [tpot2.estimator_objective_functions.number_of_nodes_objective],
                        #                 'other_objective_functions_weights':[-1],
                                        
                        #                 'memory_limit':'20GB', 
                        #                 'preprocessing':False,
                        #     },
                        #     },



                        # {
                        # 'automl': tpot2.TPOTClassifier,
                        # 'exp_name' : 'tpot2_untimed_30_gen_roc_auc_symbolic',
                        # 'params': {
                        #             'scorers' : ['roc_auc'],
                        #             'population_size' : 240, 
                        #             'generations' : 1000, 
                        #             'n_jobs':n_jobs,

                        #             'inner_config_dict' : ['arithmetic_transformer'],
                        #             'leaf_config_dict' : ['feature_set_selector'],
                        #             'root_config_dict' : {LogisticRegression: params_LogisticRegression},

                        #             'survival_percentage' : 0.5,
                        #             'crossover_probability' : .3,
                        #             'mutate_probability' : .4,
                        #             'mutate_then_crossover_probability': 0,
                        #             'crossover_then_mutate_probability': .3,
    
                        #             'verbose':5, 
                        #             'max_time_seconds':60*60,
                        #             'max_eval_time_seconds':60*6, 

                        #             'other_objective_functions' : [tpot2.estimator_objective_functions.average_path_length_objective],
                        #             'other_objective_functions_weights':[-1],
                                    
                        #             'memory_limit':'20GB', 
                        #             'preprocessing':False,
                        #             },
                        # },
    ]
        
    task_id_lists = [189865,
                    167200,
                    126026,
                    189860,
                    75127,
                    189862,
                    75105,
                    168798,
                    126029,
                    168796,
                    167190,
                    189866,
                    167104,
                    167083,
                    167184,
                    126025,
                    75097, 
                    167181,
                    168797,
                    189861,
                    167161,
                    167149,
                    ]
    
    utils.loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs)


if __name__ == '__main__':
    main()