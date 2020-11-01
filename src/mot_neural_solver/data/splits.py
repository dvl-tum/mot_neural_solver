_SPLITS = {}

#################
# MOT15
#################

# sequences used for training
mot15_train_seqs = ['KITTI-17', 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Stadtmitte']

# Additional train sequences not used for tranining (since they are present in MOT17 etc.)
add_mot15_train_seqs = ['ETH-Pedcross2', 'TUD-Campus', 'KITTI-13', 'Venice-2', 'ADL-Rundle-8', 'ADL-Rundle-6']
_SPLITS['mot15_train_gt'] = {'2DMOT2015/train': [f'{seq}-GT' for seq in mot15_train_seqs]}
_SPLITS['mot15_train'] = {'2DMOT2015/train': mot15_train_seqs + add_mot15_train_seqs}

# Test sequences
test_seqs =  ['TUD-Crossing', 'PETS09-S2L2', 'ETH-Jelmoli', 'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre',
                  'ADL-Rundle-1', 'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1']
_SPLITS['mot15_test'] = {'2DMOT2015/test': test_seqs}


#################
# MOT17
#################
dets = ('DPM', 'FRCNN', 'SDP')

# Train sequences:
train_seq_nums=  (2, 4, 5, 9, 10, 11, 13)
_SPLITS['mot17_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mot17_train'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in train_seq_nums for det in dets]}
_SPLITS['mot17_train_sdp'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-SDP' for seq_num in train_seq_nums ]}


# Cross Validation splits
_SPLITS['mot17_split_1_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 5, 9, 10, 13)]}
_SPLITS['split_1_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in dets]}

_SPLITS['mot17_split_2_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 11, 10, 13)]}
_SPLITS['split_2_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in dets]}

_SPLITS['mot17_split_3_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (4, 5, 9, 11)]}
_SPLITS['split_3_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in dets]}

_SPLITS['debug'] = {'MOT17Labels/train': ['MOT17-02-FRCNN']}


# Test sequences
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)
_SPLITS['mot17_test'] = {'MOT17Labels/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in test_seq_nums for det in dets]}

############
# MOT20
############

train_seq_nums=  (1, 2, 3, 5)
# Train / Val sequences
_SPLITS['mot20_train'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in train_seq_nums]}
_SPLITS['mot20_train_gt'] = {'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mot20_train_wo_val'] = {'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in (1, 2,  5)]}
_SPLITS['mot20_val'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (3,)]}

_SPLITS['mot20_train_gt+'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 9)],
                              'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums]}


# Cross-Val
for split_num, val_seq in enumerate(train_seq_nums, 1):
    _SPLITS[f'mot20_train_{split_num}'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 9)],
                                'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums if seq_num != val_seq]}
    _SPLITS[f'mot20_val_{split_num}'] = {'MOT20/train': [f'MOT20-{val_seq:02}']}



# Test Sequences
_SPLITS['mot20_test'] = {'MOT20/test': [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]}

# Combinations:
_SPLITS['all_train'] = {**_SPLITS['mot17_train_gt'], **_SPLITS['mot15_train_gt'], **_SPLITS['mot20_train_gt']}
_SPLITS['all_test'] = {**_SPLITS['mot17_test'], **_SPLITS['mot15_test'], **_SPLITS['mot20_test']}




