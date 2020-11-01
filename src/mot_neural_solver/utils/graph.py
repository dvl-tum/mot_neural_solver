import numpy as np

import torch
from torch_scatter import scatter_mean


def get_time_valid_conn_ixs(frame_num, max_frame_dist, return_undirected = True, use_cuda = False):
    """
    Determines the valid connections among nodes (detections) according to their time distance. Valid connections
    are those for which nodes are not in the same frame, and their time dist is not greater than max_frame_dist.
    (Does the same as the function above at similar speed but does not require as much memory)
    Args:
        frame_num: np.array with shape (num_nodes,), indicating the frame number of each node.
        max_frame_dist: maximum distance allowed among detections (in number of frames) (if 'max', it is ignored)
        use_cuda: bool indicates if operation must be performed in GPU
        return_undirected: bool, determines whether both (i,j) and (j, i) is returned for each edge

    Returns:
        torch.Tensor with shape (2, num_edges) corresponding to the valid edges

    """
    assert isinstance(max_frame_dist, (int, np.uint)) or max_frame_dist == 'max'

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    frame_num = frame_num.to(device)
    assert (torch.sort(frame_num)[0] == frame_num).all(), "Detections are NOT sorted by timestamp. Graph was not created properly!"

    changepoints = torch.where(frame_num[1:] != frame_num[:-1])[0] + 1
    changepoints = torch.cat((changepoints, torch.as_tensor([frame_num.shape[0]]).to(changepoints.device)))
    all_det_ixs = torch.arange(frame_num.shape[0], device=frame_num.device)

    edge_ixs = []
    for start_frame_ix, end_frame_ix in zip(changepoints[:-1], changepoints[1:]):
        curr_frame_ixs = all_det_ixs[start_frame_ix: end_frame_ix]
        curr_frame_num = frame_num[curr_frame_ixs[0]]

        # Update past frames
        if max_frame_dist != 'max':
            past_frames_ixs = torch.where(torch.abs(frame_num[:start_frame_ix] - curr_frame_num) <= max_frame_dist)[0]

        else:
            past_frames_ixs = all_det_ixs[:start_frame_ix]

        edge_ixs.append(torch.cartesian_prod(past_frames_ixs, curr_frame_ixs))

    edge_ixs = torch.cat(edge_ixs).T

    if return_undirected:
        return edge_ixs

    else:
        return torch.cat((edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1)



def get_time_valid_conn_ixs_old(frame_num, max_frame_dist, use_cuda, return_undirected = True):
    """
    Determines the valid connections among nodes (detections) according to their time distance. Valid connections
    are those for which nodes are not in the same frame, and their time dist is not greater than max_frame_dist.
    Args:
        frame_num: np.array with shape (num_nodes,), indicating the frame number of each node.
        max_frame_dist: maximum distance allowed among detections (in number of frames) (if 'max', it is ignored)
        use_cuda: bool indicates if operation must be performed in GPU
        return_undirected: bool, determines whether both (i,j) and (j, i) is returned for each edge

    Returns:
        torch.Tensor with shape (2, num_edges) corresponding to the valid edges

    """
    assert isinstance(max_frame_dist, (int, np.uint)) or max_frame_dist == 'max'

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    frame_num = frame_num.to(device)
    frames_time_dists = torch.abs(frame_num.reshape(-1, 1) - frame_num.reshape(1, -1))
    frame_dist_cond = frames_time_dists > 0

    if max_frame_dist != 'max':
        frame_dist_cond = frame_dist_cond & (frames_time_dists <= max_frame_dist)

    row, col = torch.where(frame_dist_cond)
    if not return_undirected:
        return row, col

    mask = row < col
    row, col = row[mask], col[mask]

    return torch.stack((row, col)).cpu()


def get_knn_mask(pwise_dist, edge_ixs, num_nodes, top_k_nns, use_cuda, reciprocal_k_nns=False, symmetric_edges = True ):
    """
    Determines the edge indices corresponding to the KNN graph according to pruning_out
    Args:
        pwise_dist: Distance for each edge. Smaller --> closer
        graph_obj: Graph Object containing edge ixs, etc.
        top_k_nns: Number of K NNs to have per each node
        reciprocal_k_nns:  Determines whether NNs relationships need to be reciprocal
        symmetric_edges: Indicates whether edge_ixs contains edges in both directions or not

    Returns:
        torch bool tensor with shape (num_edges,). For each position (edge) True --> keep. False --> prune

    """
    # Construct a matrix with distance scores
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    dist_mat = torch.empty((num_nodes, num_nodes), device=device)
    dist_mat[...] = np.inf
    dist_mat[edge_ixs[0], edge_ixs[1]] = pwise_dist.to(device).view(-1)
    if not symmetric_edges:
        dist_mat[edge_ixs[1], edge_ixs[0]] = pwise_dist.to(device).view(-1)

    row, col = torch.from_numpy(np.indices((num_nodes, num_nodes))).to(device) # In both cases we need these indices

    # For each node, order its Neighboring nodes and select the top K
    per_node_order = torch.argsort(dist_mat, dim=1, descending=False)

    # Now, build masks to determine the final 'pruned edges mask'
    # Build a matrix where each entry represents the 'ranking' position that the given node (row) has for the corresponding (col)
    ranking_mat = torch.zeros_like(dist_mat).long()
    ranking_mat[row.view(-1), per_node_order.view(-1)] = col.view(-1)

    # If we are using 'reciprocal' NNs, make sure that both (i, j) and (j, i) are among each other's NNs
    in_k_nns = ranking_mat < top_k_nns
    if reciprocal_k_nns:
        in_k_nns = in_k_nns & in_k_nns.T

    else:
        in_k_nns =  in_k_nns | in_k_nns.T

    # Make sure that the edges we use were in the original set (i.e not set to inf)
    is_feasible = dist_mat != -float('inf')
    knns_conds = in_k_nns & is_feasible

    # Now, Gather the final mask
    pruned_mask = knns_conds[edge_ixs[0], edge_ixs[1]]

    return pruned_mask


def compute_edge_feats_dict(edge_ixs, det_df, fps, use_cuda):
    """
    Computes a dictionary of edge features among pairs of detections
    Args:
        edge_ixs: Edges tensor with shape (2, num_edges)
        det_df: processed detections datafrmae
        fps: fps for the given sequence
        use_cuda: bool, determines whether operations must be performed in GPU
    Returns:
        Dict where edge key is a string referring to the attr name, and each val is a tensor of shape (num_edges)
        with vals of that attribute for each edge.

    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    row, col = edge_ixs

    secs_time_dists = torch.from_numpy(det_df['frame'].values).float().to(device) / fps

    bb_height = torch.from_numpy(det_df['bb_height'].values).float().to(device)
    bb_width = torch.from_numpy(det_df['bb_width'].values).float().to(device)

    feet_x = torch.from_numpy(det_df['feet_x'].values).float().to(device)
    feet_y = torch.from_numpy(det_df['feet_y'].values).float().to(device)

    mean_bb_heights = (bb_height[row] + bb_height[col]) / 2

    edge_feats_dict = {'secs_time_dists': secs_time_dists[col] - secs_time_dists[row],

                       'norm_feet_x_dists': (feet_x[col] - feet_x[row]) / mean_bb_heights,
                       'norm_feet_y_dists': (feet_y[col] - feet_y[row]) / mean_bb_heights,

                       'bb_height_dists': torch.log(bb_height[col] / bb_height[row]),
                       'bb_width_dists': torch.log(bb_width[col] / bb_width[row])}

    return edge_feats_dict


def to_undirected_graph(mot_graph, attrs_to_update = ('edge_preds', 'edge_labels')):
    """
    Given a MOTGraph object, it updates its Graph object to make its edges directed (instead of having each edge
    (i, j) appear twice (e.g. (i, j) and (j, i)) it only keeps (i, j) with i <j)
    It averages edge attributes in attrs_to_update accordingly.

    Args:
        mot_graph: MOTGraph object
        attrs_to_update: list/tuple of edge attribute names, that will be averaged over each pair of directed edges
    """

    # Make edges undirected
    sorted_edges, _ = torch.sort(mot_graph.graph_obj.edge_index, dim=0)
    undirected_edges, orig_indices = torch.unique(sorted_edges, return_inverse=True, dim=1)
    assert sorted_edges.shape[1] == 2 * undirected_edges.shape[1], "Some edges were not duplicated"
    mot_graph.graph_obj.edge_index = undirected_edges

    # Average values between each pair of directed edges for all attributes in 'attrs_to_update'
    for attr_name in attrs_to_update:
        if hasattr(mot_graph.graph_obj, attr_name):
            undirected_attr = scatter_mean(getattr(mot_graph.graph_obj, attr_name), orig_indices)
            setattr(mot_graph.graph_obj, attr_name, undirected_attr)

def to_lightweight_graph(mot_graph, attrs_to_del=('reid_emb_dists', 'x', 'edge_attr', 'edge_labels')):
    """
    Deletes attributes in mot_graph that are not needed for inference, to save memory
    Args:
        mot_graph: MOTGraph object
        attrs_to_del: tuple/list of attributes to delete

    """
    mot_graph.graph_obj.num_nodes = mot_graph.graph_obj.num_nodes
    mot_graph.graph_obj.node_names = torch.arange(mot_graph.graph_obj.num_nodes).to(mot_graph.graph_obj.device())

    # Delete attributes that are unnecessary for inference
    for attr_name in attrs_to_del:
        if hasattr(mot_graph.graph_obj, attr_name):
            delattr(mot_graph.graph_obj, attr_name)

    # Prune edges with low prediction score
    edges_mask = mot_graph.graph_obj.edge_preds >= 0.5
    mot_graph.graph_obj.edge_index = mot_graph.graph_obj.edge_index.T[edges_mask].T
    mot_graph.graph_obj.edge_preds = mot_graph.graph_obj.edge_preds[edges_mask]