import tpot2
import sklearn.metrics
import sklearn
import argparse
import tpot
import utils
import tpot2
import sklearn.datasets


def main():
    # Read in arguements
    parser = argparse.ArgumentParser()

    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="multiclass_results", required=False, nargs='?')

    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = 48
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    experiments = [

                                                                                                                                {
                        'automl': tpot2.TPOTEstimator,
                        'exp_name' : 'tpot2_untimed_30_gen_neg_log_loss',
                        'params': {
                    
                        'scorers' :     ['neg_log_loss'],   
                        'scorers_weights': [1] ,
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
                                        
                                        'other_objective_functions' : [tpot2.objectives.number_of_nodes_objective],
                                        'other_objective_functions_weights':[-1],
                                        
                                        'memory_limit':"20GB",
                                        'preprocessing':False,
                                        'classification':True,
                            },
                            },

                        {
                        'automl': tpot.TPOTClassifier,
                        'exp_name' : 'tpot_untimed_30_gen_neg_log_loss',
                        'params': {
                                    'scoring': 'neg_log_loss',
                                    'population_size' : 48, 
                                    'generations' : 29, 
                                    'n_jobs':n_jobs,
                                    'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                    'verbosity': 2, 
                                    'max_time_mins': None,
                                    'max_eval_time_mins' : 5,         
                        },
                        },


    ]
        
    task_id_lists = [
                            168794,
                            189873,
                            167152,
                            189909,
                            189872,
                            168795, #This one has fewer than 10 instances of a class
                            167185,
                            189905,
                            189874,
                            75193,
                            167168,
                            168793,
                            189906,
                            167201,
                            189908,
                            168792,
                            189871
                            ]
    
    utils.loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs)


if __name__ == '__main__':
    main()