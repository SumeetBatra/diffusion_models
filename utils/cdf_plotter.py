from utils.metric_calculator import reevaluate_ppga_archive_mem, compile_cdf, make_cdf_plot, index_of, archive_df_to_archive
from algorithm.train_autoencoder import shaped_elites_dataset_factory
from attrdict import AttrDict
from utils.brax_utils import shared_params
import numpy as np
import copy
import pandas as pd
import os
import matplotlib
from envs.brax_custom import reward_offset
matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 20,
    }
)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# matplotlib.rcParams.update(matplotlib.rcParamsDefault)

if __name__ == '__main__':

    envs = [("halfcheetah", False), ("ant", True), ("walker2d", False), ("humanoid", True),] #["halfcheetah", "ant", "walker2d", "humanoid"]



    train_batch_size = 32
    Num = 5000

    for env, centering in envs:
        base_cfg = AttrDict(shared_params[env])
        base_cfg['title'] = env

        cfg = copy.copy(base_cfg)
        cfg.update({'archive_dir': None, 'algorithm': "policy_diffusion"})    
        
        # check if original archive exists
        existing_original_archive = f'/home/shashank/research/qd/paper_results2_without_averaging/{env}/diffusion_model/cdf_original.csv'
        if os.path.exists(existing_original_archive):
            orig_cdf = pd.read_csv(existing_original_archive)
            print('Original Archive Exists')
        
        else:
            weight_normalizer = None
            train_dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(
                env, batch_size=train_batch_size, is_eval=False,
                center_data=centering,
                use_language=False,
                N=Num,
                weight_normalizer=weight_normalizer)
            
            env_cfg = AttrDict(shared_params[env]['env_cfg'])
            env_cfg.seed = 1111
            env_cfg.clip_obs_rew = True

            archive_df = train_archive[0]

            soln_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
            archive_dims = [env_cfg['grid_size']] * env_cfg['num_dims']
            ranges = [(0.0, 1.0)] * env_cfg['num_dims']

            original_archive = archive_df_to_archive(
                archive_df,
                solution_dim=soln_dim,
                dims=archive_dims,
                ranges=ranges,
                seed=env_cfg.seed,
                qd_offset=reward_offset[env])

            print('Re-evaluated Original Archive')
            archive_info = reevaluate_ppga_archive_mem(
                env_cfg,
                True,
                False,
                original_archive,
                average=False)

            orig_cdf = compile_cdf(cfg, dataframes=[archive_info["Archive"]])

            # save the orig_cdf dataframe to csv
            orig_cdf.to_csv(f'/home/shashank/research/qd/paper_results2_without_averaging/{env}/diffusion_model/cdf_original.csv', index=False)

        # load the algo_cdf dataframe from csv
        algo_cdf = pd.read_csv(f'/home/shashank/research/qd/paper_results2_without_averaging/{env}/diffusion_model/cdf_{centering}.csv')
        
        env_idx = index_of(env)
        fig, axs = matplotlib.pyplot.subplots(1, 1, figsize=(6, 6))
        
        make_cdf_plot(cfg, algo_cdf, axs, original=False, standalone=True)
        make_cdf_plot(cfg, orig_cdf, axs, original=True, standalone=True)



        # save the figure
        fig.savefig(f'/home/shashank/research/qd/paper_results2_without_averaging/{env}/diffusion_model/new_cdf_{centering}.png')