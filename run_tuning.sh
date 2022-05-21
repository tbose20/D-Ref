for INP_per in 0.1 0.2 0.3 0.4
do
        for alp in 0.1 0.5 1.0 10.0 20.0 30.0 40.0 50.0 60.0
        do

                python -u train_eval_bc.py -dataset HatEval -out_dataset  Dynamic -encoder bert -data_dir tasks/ -model_dir models/ -alpha ${alp} -perc_inp ${INP_per}
                rm -r models/HatEval/bert_dot0.model
                rm -r models/HatEval/bert_dot_predictive_performances.csv
                rm -r models/HatEval/model_run_stats
        done
done

