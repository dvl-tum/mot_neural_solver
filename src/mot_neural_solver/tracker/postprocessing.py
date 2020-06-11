import numpy as np
import pandas as pd

class Postprocessor:
    """
    Class used   to perform all postprocessing.
    """
    def __init__(self, traj_df, seq_info_dict, eval_params):
        self.traj_df = traj_df.rename(columns={'id': 'ped_id'})
        self.eval_params = eval_params
        self.seq_info_dict = seq_info_dict

    def drop_short_trajectories(self):
        # Discard trajectories that appear few times
        short_tracklets_ix = self.traj_df.index[
            self.traj_df.groupby('ped_id')['frame'].transform('count') < self.eval_params['min_track_len']]
        self.traj_df.drop(short_tracklets_ix, inplace=True)

    def interpolate_trajectories(self):
        # Add bbox center coords
        self.traj_df['mid_x'] = self.traj_df['bb_left'] + 0.5 * self.traj_df['bb_width']
        self.traj_df['mid_y'] = self.traj_df['bb_top'] + 0.5 * self.traj_df['bb_height']

        # Build sub_dfs with full trajectories across all missing frames
        reixed_traj_df = self.traj_df.set_index('ped_id')
        full_traj_dfs = []
        traj_start_ends = self.traj_df.groupby('ped_id')['frame'].agg(['min', 'max'])
        for ped_id, (traj_start, traj_end) in traj_start_ends.iterrows():
            if ped_id != -1:
                full_traj_df = pd.DataFrame(data=np.arange(traj_start, traj_end + 1), columns=['frame'])
                partial_traj_df = reixed_traj_df.loc[ped_id].reset_index()

                # Interpolate bb centers, heights and widths
                full_traj_df = pd.merge(full_traj_df,
                                        partial_traj_df[['ped_id', 'frame', 'mid_x', 'mid_y', 'bb_height', 'bb_width']],
                                        how='left', on='frame')
                full_traj_df = full_traj_df.sort_values(by='frame').interpolate()
                full_traj_dfs.append(full_traj_df)

        self.traj_df = pd.concat(full_traj_dfs)

        # Recompute bb coords based on the interpolated centers, heights and widths
        self.traj_df['bb_left'] = self.traj_df['mid_x'] - 0.5 * self.traj_df['bb_width']
        self.traj_df['bb_top'] = self.traj_df['mid_y'] - 0.5 * self.traj_df['bb_height']
        self.traj_df['bb_right'] = self.traj_df['mid_x'] + 0.5 * self.traj_df['bb_width']
        self.traj_df['bb_bot'] = self.traj_df['mid_y'] + 0.5 * self.traj_df['bb_height']

    def postprocess_trajectories(self):
        self.drop_short_trajectories()
        self.interpolate_trajectories()

        return self.traj_df