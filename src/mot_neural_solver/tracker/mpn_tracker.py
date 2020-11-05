import numpy as np
import pandas as pd

import torch

from mot_neural_solver.data.mot_graph import Graph

from mot_neural_solver.tracker.projectors import GreedyProjector, ExactProjector
from mot_neural_solver.tracker.postprocessing import Postprocessor

from mot_neural_solver.utils.graph import get_knn_mask, to_undirected_graph, to_lightweight_graph

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from lapsolver import solve_dense

VIDEO_COLUMNS = ['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right', 'bb_bot']
TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']


class MPNTracker:
    """
    Class used to track video sequences.

    See 'track'  method for an overview.
    """

    def __init__(self, dataset, graph_model, use_gt, eval_params=None,
                 dataset_params=None, logger=None):

        self.dataset = dataset
        self.use_gt = use_gt
        self.logger = logger

        self.eval_params = eval_params
        self.dataset_params = dataset_params

        self.graph_model = graph_model

        if self.graph_model is not None:
            self.graph_model.eval()

    def _estimate_frames_per_graph(self, seq_name):
        """
        Determines the number of frames to be included in each batch of frames evaluated within a sequence
        """
        num_frames = len(self.dataset.seq_det_dfs[seq_name].frame.unique())
        num_detects = self.dataset.seq_det_dfs[seq_name].shape[0]

        avg_detects_per_frame = num_detects / float(num_frames)
        expected_frames_per_graph = round(self.dataset.dataset_params['max_detects'] / avg_detects_per_frame)

        return min(expected_frames_per_graph, self.dataset.dataset_params['frames_per_graph'])

    def _estimate_max_frame_dist(self, seq_name, frames_per_graph):
        step_size = self.dataset.seq_info_dicts[seq_name]['step_size']
        # frames_per_graph = self._estimate_frames_per_graph(seq_name)

        # TODO: Should use seconds as unit, and not number of frames
        if self.dataset.dataset_params['max_frame_dist'] == 'max':
            max_frame_dist = step_size * (frames_per_graph - 1)

        else:
            max_frame_dist = self.dataset.dataset_params['max_frame_dist']

        return max_frame_dist

    def _load_full_seq_graph_object(self, seq_name, start_frame, end_frame, max_frame_dist):
        """
        Loads a MOTGraph (see data/mot_graph.py) object corresponding to the entire sequence.
        """

        full_graph = self.dataset.get_from_frame_and_seq(seq_name=seq_name,
                                                         start_frame=start_frame,
                                                         end_frame=end_frame,
                                                         return_full_object=True,
                                                         ensure_end_is_in=True,
                                                         max_frame_dist=max_frame_dist,
                                                         inference_mode=True)
        # full_graph.frames_per_graph = frames_per_graph
        return full_graph

    def _predict_edges(self, subgraph):
        """
        Predicts edge values for a subgraph (i.e. batch of frames) from the entire sequence.
        Args:
            subgraph: Graph Object corresponding to a subset of frames

        Returns:
            tuple containing a torch.Tensor with the predicted value for every edge in the subgraph, and a binary mask
            indicating which edges inside the subgraph where pruned with KNN
        """
        # Prune graph edges
        knn_mask = get_knn_mask(pwise_dist=subgraph.reid_emb_dists, edge_ixs=subgraph.edge_index,
                                num_nodes=subgraph.num_nodes, top_k_nns=self.dataset_params['top_k_nns'],
                                use_cuda=True, reciprocal_k_nns=self.dataset_params['reciprocal_k_nns'],
                                symmetric_edges=True)
        subgraph.edge_index = subgraph.edge_index.T[knn_mask].T
        subgraph.edge_attr = subgraph.edge_attr[knn_mask]
        if hasattr(subgraph, 'edge_labels'):
            subgraph.edge_labels = subgraph.edge_labels[knn_mask]

        # Predict active edges
        if self.use_gt:  # For debugging purposes and obtaining oracle results
            pruned_edge_preds = subgraph.edge_labels

        else:
            with torch.no_grad():
                pruned_edge_preds = torch.sigmoid(self.graph_model(subgraph)['classified_edges'][-1].view(-1))

        edge_preds = torch.zeros(knn_mask.shape[0]).to(pruned_edge_preds.device)
        edge_preds[knn_mask] = pruned_edge_preds

        if self.eval_params['set_pruned_edges_to_inactive']:
            return edge_preds, torch.ones_like(knn_mask)

        else:
            return edge_preds, knn_mask  # In this case, pruning an edge counts as not predicting a value for it at all
            # However, if it is pruned for every batch, then it counts as inactive.

    def _evaluate_graph_in_batches(self, subseq_graph, frames_per_graph):
        """
        Feeds the entire sequence though the MPN in batches. It does so by applying a 'sliding window' over the sequence,
        where windows correspond consecutive pairs of start/end frame locations (e.g. frame 1 to 15, 5 to 20, 10 to 25,
        etc.).
        For every window, a subgraph is created by selecting all detections that fall inside it. Then this graph
        is fed to the message passing network, and predictions are stored.
        Since windows overlap, we end up with several predictions per edge. We simply average them overall all
        windows.
        """
        device = torch.device('cuda')
        all_frames = np.array(subseq_graph.frames)
        frame_num_per_node = torch.from_numpy(subseq_graph.graph_df.frame.values).to(device)
        node_names = torch.arange(subseq_graph.graph_obj.x.shape[0])

        # Iterate over overlapping windows of (starg_frame, end_frame)
        overall_edge_preds = torch.zeros(subseq_graph.graph_obj.num_edges).to(device)
        overall_num_preds = overall_edge_preds.clone()
        for eval_round, (start_frame, end_frame) in enumerate(zip(all_frames, all_frames[frames_per_graph - 1:])):
            assert ((start_frame <= all_frames) & (all_frames <= end_frame)).sum() == frames_per_graph

            # Create and evaluate a a subgraph corresponding to a batch of frames
            nodes_mask = (start_frame <= frame_num_per_node) & (frame_num_per_node <= end_frame)
            edges_mask = nodes_mask[subseq_graph.graph_obj.edge_index[0]] & nodes_mask[
                subseq_graph.graph_obj.edge_index[1]]

            subgraph = Graph(x=subseq_graph.graph_obj.x[nodes_mask],
                             edge_attr=subseq_graph.graph_obj.edge_attr[edges_mask],
                             reid_emb_dists=subseq_graph.graph_obj.reid_emb_dists[edges_mask],
                             edge_index=subseq_graph.graph_obj.edge_index.T[edges_mask].T - node_names[nodes_mask][0])

            if hasattr(subseq_graph.graph_obj, 'edge_labels'):
                subgraph.edge_labels = subseq_graph.graph_obj.edge_labels[edges_mask]

            # Predict edge values for the current batch
            edge_preds, pred_mask = self._predict_edges(subgraph=subgraph)

            # Store predictions
            overall_edge_preds[edges_mask] += edge_preds
            assert (overall_num_preds[torch.where(edges_mask)[0][pred_mask]] == overall_num_preds[edges_mask][
                pred_mask]).all()
            overall_num_preds[torch.where(edges_mask)[0][pred_mask]] += 1

        # Average edge predictions over all batches, and over each pair of directed edges
        final_edge_preds = overall_edge_preds / overall_num_preds
        final_edge_preds[torch.isnan(final_edge_preds)] = 0
        subseq_graph.graph_obj.edge_preds = final_edge_preds
        to_undirected_graph(subseq_graph, attrs_to_update=('edge_preds', 'edge_labels'))
        to_lightweight_graph(subseq_graph)
        return subseq_graph
        # print(time() - t)

    def _project_graph_model_output(self, subseq_graph):
        """
        Rounds MPN predictions either via Linear Programming or a greedy heuristic
        """
        if self.eval_params['rounding_method'] == 'greedy':
            projector = GreedyProjector(subseq_graph)

        elif self.eval_params['rounding_method'] == 'exact':
            projector = ExactProjector(subseq_graph, solver_backend=self.eval_params['solver_backend'])

        else:
            raise RuntimeError("Rounding type for projector not understood")

        projector.project()

        subseq_graph.graph_obj = subseq_graph.graph_obj.numpy()
        subseq_graph.constr_satisf_rate = projector.constr_satisf_rate

        return subseq_graph

    def _assign_ped_ids(self, subseq_graph):
        """
        Assigns pedestrian Ids to each detection in the sequence, by determining all connected components in the graph
        """
        # Only keep the non-zero edges and Express the result as a CSR matrix so that it can be fed to 'connected_components')
        nonzero_mask = subseq_graph.graph_obj.edge_preds == 1
        nonzero_edge_index = subseq_graph.graph_obj.edge_index.T[nonzero_mask].T
        nonzero_edges = subseq_graph.graph_obj.edge_preds[nonzero_mask].astype(int)
        graph_shape = (subseq_graph.graph_obj.num_nodes, subseq_graph.graph_obj.num_nodes)
        csr_graph = csr_matrix((nonzero_edges, (tuple(nonzero_edge_index))), shape=graph_shape)

        # Get the connected Components:
        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
        assert len(labels) == subseq_graph.graph_df.shape[0], "Ped Ids Label format is wrong"

        # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
        final_projected_output = subseq_graph.graph_df.copy()
        final_projected_output['ped_id'] = labels
        final_projected_output = final_projected_output[VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

        return final_projected_output

    def _determine_seq_cutpoints(self, seq_name):
        """
        In order to reduce the memory requirements of processing large sequences (i.e. avoid storing the entire graph
        corresponding to the sequence), we split them into smaller sequences, get the tracking output of each of them,
        and then merge the results.

        """
        seq_det_df = self.dataset.seq_det_dfs[seq_name]
        step_size = self.dataset.seq_info_dicts[seq_name]['step_size']
        start_frame, end_frame = seq_det_df.frame.min(), seq_det_df.frame.max()

        valid_frames = np.arange(start_frame, end_frame + 1, step_size)
        if end_frame not in valid_frames:
            valid_frames = valid_frames.tolist() + [end_frame]
        seq_det_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()

        assert_str = f"{self.eval_params['max_dets_per_graph_seq']} to be larger than than twice the maximum amount of dets possible to have in one frame"
        assert self.eval_params['max_dets_per_graph_seq'] > 2 * seq_det_df.groupby('frame')[
            'bb_left'].count().max(), assert_str

        num_dets = seq_det_df.shape[0]
        frame_cutpoints = [seq_det_df.frame.iloc[i] for i in
                           np.arange(0, num_dets, self.eval_params['max_dets_per_graph_seq'])]
        frame_cutpoints += [seq_det_df.frame.iloc[-1] + 1]
        assert len(frame_cutpoints) == len(set(frame_cutpoints))

        return frame_cutpoints

    def _merge_subseq_dfs(self, subseq_dfs):
        seq_df = subseq_dfs[0]
        for subseq_df in subseq_dfs[1:]:
            # Make sure that ped_ids in subseq_df are new and all greater than the ones in seq_df:
            subseq_df['ped_id'] += seq_df['ped_id'].max() + 1

            """
            # Match ids from both sides
            # TODO: We could do this based on bipartite matching based on common detection_ids per ped_id in the
            # TODO: overlap section
            left_ped_ids = seq_df.groupby('ped_id')['detection_id'].max().reset_index().rename(
                columns={'ped_id': 'left_ped_id'})
            right_ped_ids = subseq_df.groupby('ped_id')['detection_id'].min().reset_index().rename(
                columns={'ped_id': 'right_ped_id'})
            matched_ids = pd.merge(left_ped_ids, right_ped_ids, on='detection_id', how='inner')[
                ['left_ped_id', 'right_ped_id']]
            

            """
            intersect_frames = np.intersect1d(seq_df.frame, subseq_df.frame)
            left_df = seq_df[['detection_id', 'ped_id']][seq_df.frame.isin(intersect_frames)]
            left_ids_pos = left_df[['ped_id']].drop_duplicates(); left_ids_pos['ped_id_pos'] = np.arange(left_ids_pos.shape[0])
            left_df = left_df.merge(left_ids_pos, on='ped_id').set_index('detection_id')

            #left_df['left_ped_id_pos'] = np.arange(left_df.shape[0])
            right_df = subseq_df[['detection_id', 'ped_id']][subseq_df.frame.isin(intersect_frames)]
            right_ids_pos = right_df[['ped_id']].drop_duplicates(); right_ids_pos['ped_id_pos'] = np.arange(right_ids_pos.shape[0])
            right_df = right_df.merge(right_ids_pos, on='ped_id').set_index('detection_id')


            #right_df['right_ped_id_pos'] = np.arange(right_df.shape[0])
            # Build the linear assignment matrix
            common_boxes = left_df[['ped_id_pos']].join(right_df['ped_id_pos'], lsuffix='_left', rsuffix='_right').dropna(thresh = 2).reset_index().groupby(['ped_id_pos_left', 'ped_id_pos_right'])['detection_id'].count()
            common_boxes = common_boxes.reset_index().astype(int)

            cost_mat = np.full((common_boxes['ped_id_pos_left'].max() +1 , common_boxes['ped_id_pos_right'].max() + 1), fill_value = np.nan)
            cost_mat[common_boxes['ped_id_pos_left'].values, common_boxes['ped_id_pos_right'].values] =  - common_boxes['detection_id'].values
            matched_left_ids_pos, matched_right_ids_pos = solve_dense(cost_mat)
            matched_ids = pd.DataFrame(data = np.stack((left_ids_pos['ped_id'].values[matched_left_ids_pos],
                                                       right_ids_pos['ped_id'].values[matched_right_ids_pos])).T,
                                       columns = ['left_ped_id', 'right_ped_id'])


            # Assign the ids matched to subseq_df
            subseq_df = pd.merge(subseq_df, matched_ids, how='outer', left_on='ped_id', right_on='right_ped_id')
            subseq_df['left_ped_id'].fillna(np.inf, inplace=True)
            subseq_df['ped_id'] = np.minimum(subseq_df['left_ped_id'], subseq_df['ped_id'])

            # Update seq_df
            seq_df = pd.concat([seq_df, subseq_df[subseq_df['frame'] > seq_df['frame'].max()]])

        return seq_df

    def track(self, seq_name, output_path=None):
        """
        Main method. Given a sequence name, it tracks all detections and produces an output DataFrame, where each
        detection is assigned an ID.

        It starts loading a the graph corresponding to an entire video sequence and detections, then uses an MPN to
        sequentially evaluate batches of frames (i.e. subgraphs) and finally rounds predictions and applies
        postprocessing.

        """
        from time import time

        #print(f"Processing Seq {seq_name}")
        frames_per_graph = self._estimate_frames_per_graph(seq_name)
        max_frame_dist = self._estimate_max_frame_dist(seq_name, frames_per_graph)
        frame_cutpoints = self._determine_seq_cutpoints(seq_name)
        subseq_dfs = []
        constr_sr = 0
        for start_frame, end_frame in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            t = time()

            # Load the graph corresponding to the entire subsequence
            subseq_graph = self._load_full_seq_graph_object(seq_name, start_frame if start_frame == frame_cutpoints[0] else start_frame - max_frame_dist,
                                                            end_frame, max_frame_dist)

            # Feed graph through MPN in batches
            subseq_graph = self._evaluate_graph_in_batches(subseq_graph, frames_per_graph)

            # Round predictions and assign IDs to trajectories
            subseq_graph = self._project_graph_model_output(subseq_graph)
            constr_sr += subseq_graph.constr_satisf_rate
            subseq_df = self._assign_ped_ids(subseq_graph)
            subseq_dfs.append(subseq_df)

            del subseq_graph
            #print("Max Mem allocated:", torch.cuda.max_memory_allocated(torch.device('cuda:0'))/1e9)
            #print(f"Done with Sequence chunk, it took {time()-t}")


        constr_sr /= (len(frame_cutpoints) - 1)
        seq_df = self._merge_subseq_dfs(subseq_dfs)
        # Postprocess trajectories
        if self.eval_params['add_tracktor_detects']:

            seq_df = self._add_tracktor_detects(seq_df, seq_name)


        postprocess = Postprocessor(seq_df.copy(),
                                    seq_info_dict=self.dataset.seq_info_dicts[seq_name],
                                    eval_params=self.eval_params)

        seq_df = postprocess.postprocess_trajectories()


        if output_path is not None:
            self._save_results_to_file(seq_df, output_path)

        return seq_df, constr_sr

    def _save_results_to_file(self, seq_df, output_file_path):
        """
        Stores the tracking result to a txt file, in MOTChallenge format.
        """
        seq_df['conf'] = 1
        seq_df['x'] = -1
        seq_df['y'] = -1
        seq_df['z'] = -1

        seq_df['bb_left'] += 1  # Indexing is 1-based in the ground truth
        seq_df['bb_top'] += 1

        final_out = seq_df[TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
        final_out.to_csv(output_file_path, header=False, index=False)

    ########################################### Not revised below

    def _add_tracktor_detects(self, seq_df, seq_name):
        def ensure_detects_can_be_used(start_end_per_ped_id):
            """
            We make sure that there is no overlap between MPN trajectories. To do so, we make sure that the ending frame
            for every trajectory is smaller than the starting frame than the next one.
            """
            if start_end_per_ped_id.shape[0] == 1:  # If there is a single detection there is nothing to check
                return True

            start_end_per_ped_id_ = start_end_per_ped_id.sort_values(by='min')

            comparisons = start_end_per_ped_id_['min'].values.reshape(-1, 1) <= start_end_per_ped_id_[
                'max'].values.reshape(1, -1)
            triu_ixs, tril_ixs = np.triu_indices_from(comparisons), np.tril_indices_from(comparisons, k=-1)
            return (comparisons[triu_ixs]).all() & (~comparisons[tril_ixs]).all()

        # Retrieve the complete scene DataFrame
        """
        big_dets_df = self.dataset.seq_det_dfs[seq_name].copy()
        print("Before the big merge")
        print("Big dets shape", big_dets_df.shape)
        print("Small dets shape", seq_df.shape)

        #for

        complete_df = seq_df.merge(big_dets_df[
                                       ['detection_id', 'tracktor_id', 'frame', 'bb_left',
                                        'bb_top', 'bb_width', 'bb_height', 'bb_right',
                                        'bb_bot', 'frame_path']], how='outer')
        print("After the big merge")
        """
        big_dets_df = self.dataset.seq_det_dfs[seq_name].set_index('detection_id')
        seq_df.set_index('detection_id', inplace=True)
        #print("Before the big merge")
        #print("Big dets shape", big_dets_df.shape)
        #print("Small dets shape", seq_df.shape)
        #for

        complete_df = seq_df[['ped_id', 'conf']].join(big_dets_df[
                                       ['tracktor_id', 'frame', 'bb_left',
                                        'bb_top', 'bb_width', 'bb_height', 'bb_right',
                                        'bb_bot', 'frame_path']], how='outer').reset_index()
        #print("After the big merge")

        assert complete_df.shape[0] == big_dets_df.shape[0], "Merging to add tracktor detects did not work properly"
        unique_tracktor_ids = complete_df.tracktor_id.unique()
        complete_df.sort_values(by=['tracktor_id', 'frame'], inplace=True)
        complete_df.set_index('tracktor_id', inplace=True)
        #print("Before for Loop ok")
        for tracktor_id in unique_tracktor_ids:
            detects_per_tracktor_id = complete_df.loc[tracktor_id][['detection_id', 'ped_id', 'frame']]

            if not isinstance(detects_per_tracktor_id,
                              pd.Series):  # If there is a single detect, then there's nothing to do

                initial_num_of_dets = detects_per_tracktor_id['ped_id'].isnull().sum()
                # For each MPN id, determine which detections under this 'tracktor id
                start_end_per_ped_id = \
                    detects_per_tracktor_id[detects_per_tracktor_id.ped_id.notnull()].groupby(['ped_id'])[
                        'frame'].agg(
                        ['min', 'max'])
                # Good ONe
                # Make sure we will not mess up thnigs
                if ensure_detects_can_be_used(start_end_per_ped_id):
                    #print("Start first if")
                    # We will build an empty assignment array, to give tracktor detects their id
                    ped_ids = np.empty(detects_per_tracktor_id.shape[0])
                    ped_ids[...] = np.nan
                    for ped_id, (start_frame, end_frame) in start_end_per_ped_id.iterrows():
                        ixs = np.where(detects_per_tracktor_id['frame'].between(start_frame, end_frame))[0]
                        ped_ids[ixs] = ped_id

                    # We may want to complete our trajectories with beginning/end trajectories corresponding to tracktor.
                    # This can be crucial to save our isolated detections, and also can help compensate for using low target_fps's
                    if self.eval_params['use_tracktor_start_ends']:
                        assigned_ped_ids_ixs = np.where(~np.isnan(ped_ids))[0]
                        if len(assigned_ped_ids_ixs) > 0:
                            first_ped_id_ix, last_ped_id_ix = assigned_ped_ids_ixs.min(), assigned_ped_ids_ixs.max()
                            ped_ids[:first_ped_id_ix + 1] = ped_ids[first_ped_id_ix]
                            ped_ids[last_ped_id_ix + 1:] = ped_ids[last_ped_id_ix]

                    # print(f"Added {(~np.isnan(ped_ids)).sum()} detections to a set of {initial_num_of_dets}")

                    # Finally, assign the ped_ids to the given
                    complete_df.loc[tracktor_id, 'ped_id'] = ped_ids.reshape(-1, 1)
                    #print("End first if")
                else:
                    # print_or_log(f"Found overlapping trajectories between tracktor and MPN. Lost {detects_per_tracktor_id.shape[0]} detects", self.logger)
                    # Here we need to be more careful, we interpolate the  intervals between Id switches
                    # Determine which locations have ped ids assigned
                    #print("Start second if")
                    assign_ped_ids_ixs = sorted(np.where(detects_per_tracktor_id.ped_id.notnull())[0])
                    assign_ped_ids = detects_per_tracktor_id.iloc[assign_ped_ids_ixs]['ped_id']
                    changes = np.where((assign_ped_ids[:-1] - assign_ped_ids[1:]) != 0)[0]
                    # build_intervals

                    # Iterate over id switches among them in order to determines which intervals can be safely interpolated
                    start_ix = assign_ped_ids_ixs[0]
                    # curr_ped_id = assign_ped_ids.iloc[start_ix]
                    curr_ped_id = assign_ped_ids.iloc[0]
                    # curr_ped_id = assign_ped_ids.iloc[0]
                    interv_dict = {ped_id: [] for ped_id in assign_ped_ids}
                    for change in changes:
                        interv_dict[curr_ped_id].append(np.arange(start_ix, assign_ped_ids_ixs[change] + 1))

                        start_ix = assign_ped_ids_ixs[change + 1]  # Next ped id appearance
                        curr_ped_id = assign_ped_ids.iloc[change + 1]

                    # Append the last interval
                    end_ix = assign_ped_ids_ixs[-1]
                    interv_dict[curr_ped_id].append(np.arange(start_ix, end_ix + 1))

                    # Create the id assignment array
                    ped_ids = np.empty(detects_per_tracktor_id.shape[0])
                    ped_ids[...] = np.nan
                    for ped_id, ixs_list in interv_dict.items():
                        if len(ixs_list) > 0:
                            all_ixs = np.concatenate(ixs_list)
                            ped_ids[all_ixs] = ped_id

                    # TODO: Repeated code.
                    if self.eval_params['use_tracktor_start_ends']:
                        if len(assign_ped_ids_ixs) > 0:
                            first_ped_id_ix, last_ped_id_ix = assign_ped_ids_ixs[0], assign_ped_ids_ixs[-1]
                            ped_ids[:first_ped_id_ix + 1] = ped_ids[first_ped_id_ix]
                            ped_ids[last_ped_id_ix + 1:] = ped_ids[last_ped_id_ix]

                    complete_df.loc[tracktor_id, 'ped_id'] = ped_ids.reshape(-1, 1)
                    # print_or_log(f"Recovered {(~np.isnan(ped_ids)).sum()} detects", self.logger)
                    #print("End second if")
        #print("Finished for loop")
        # Our final DataFrame is this one!!!!!!!!!!!!!!!!
        final_out = complete_df[complete_df.ped_id.notnull()].reset_index()
        final_out['conf'] = final_out['conf'].fillna(1)

        #print("Before groupby ok")
        # If some rare cases two dets in the same frame may get mapped to the same id, just average coordinates:
        final_out = final_out.groupby(['frame', 'frame_path', 'ped_id']).mean().reset_index()
        #print("After groupby ok")
        assert final_out[['frame', 'ped_id']].drop_duplicates().shape[0] == final_out.shape[0]
        #print("Assert ok")

        return final_out