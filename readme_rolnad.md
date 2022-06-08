# Roland

## Dataset Format

The dataset should be prepared as a `tsv` file (`csv` is also supported with only minor midification to the loader, add keyword arg `sep=','` to the `dd.read_csv` method), with the first row as column names.

Check `TODO:` 's in the `roland_generic` laoder to adapt it to your own dataset.

The following columns are required to construct a basic dynamic graph.

**Required Fields (Columns)**

* `SRC_NODE` and `DST_NODE`: unique IDs of individuals. In BSI dataset, these two columns are named as `Payer` and `Payee`
* `TIMESTAMP`: a timestamp (integer), e.g., `1230681600` denotes `2008-12-30T16:00:00`. In BSI dataset, this column is named as `Timestamp`.
* `AMOUNT`: transaction amount, this column is named as `AmountEUR` in BSI dataset.

**Optional Fields (Columns)**

* For node features, such as the country of company, add `SRC_NODECompany` and `DST_NODECompany` columns to the dataset. In BSI dataset, the company columns are `PayerCompany` and `PayeeCompany`.
* For edge features associated with transactions, such as the currency used in this transaction, simply add `Currency` column to the dataset.

## Example Dataset (BSI)

* See `./GraphGym_dev/run/datasets/bsi_synthetic.tsv` for an example of BSI dataset.

## Run Our Examples on BSI Dataset

1. Since the BSI dataset is confidential, we generated a synthetic version of it for demonstration purpose. The synthetic sample has exactly the same format as BSI dataset except it’s generated randomly, so you should **NOT** expect any algorithm to achieve any nontrivial out of sample accuracy on it. Feel free to modify the gen sample script to change the size of generated transaction graphs. **Firstly, generte a systhetic BSI dataset**, there should be one there already located at: `GraphGym_dev/run/datasets/bsi_synthetic.tsv`. You can regenerate the dataset using the following code.

    ```bash
    cd GraphGym_dev/run/datasets
    python ./syn_bsi.py
    ```

2. Use the predefined YAML and RUN files, see `/GraphGym_dev/run/run_single_example.sh`.

    Here the **ordinary recurrent GNN** denotes models based on homogenous graphs (i.e., graph without edge/node types), **complete heterogenous RGNN** contains separate networks for message types `(sender_type, edge_type, receiver_type)`, so there are `NumNodeTypes*NumEdgeTypes*NumNodeTypes` internal GNNs. The **partial heterogenous GNN** only consists of `NumNodeTypes` node feature extractors and `NumEdgeTypes` edge feature extractors.

    ```bash
    cd ./GraphGym_dev/run/
    # Ordinal recurrent GNN based on homogenous graph.
    python main.py --cfg configs/roland/examples/gnn_recurrent_example.yaml --repeat 1
    # Complete heterogenous RGNN.
    python main.py --cfg configs/roland/examples/complete_hete_example.yaml --repeat 1
    # Partial heterogenous RGNN.
    python main.py --cfg configs/roland/examples/partial_hete_example.yaml --repeat 1
    ```

## Run Models On Your Own Dataset

To deploy existing models on your own datasets, you would need to:

1. Make a copy of the generic loader at `GraphGym_dev/graphgym/contrib/loader/roland_generic.py`, modify all `TODO` in the python file to make it compatiable with your own dataset. (see section below.)

2. Create the corresponding YAML and RUN files:

    ```bash
    cd ./GraphGym_dev/run/
    python main.py --cfg YOUR_CONFIG.yaml --repeat 1
    ```

## Modify `yaml` Configuration Files

Here we provide a detailed example explaining how to modify a config yaml file. In most cases, you only need to modify a few lines to make it work on your own dataset. Here we only include fields needed to be change.

```yaml
out_dir: results
device: auto  # {'cpu', 'gpu', 'auto'}
dataset:
  format: transaction_hetero_v1  # the format needs to be compatiable with the loader.
  name: bsi_synthetic.tsv  # file name of the transaction dataset.
  is_hetero: True
  dir: /lfs/hyperturing2/0/tianyudu/GraphGym_dev/run/datasets  # dataset directory.
  task: link_pred
  shuffle: True  # must set to False to if using time series data.
  task_type: classification
  transductive: True
  split: [0.8, 0.1, 0.1]
  augment_feature: []
  augment_feature_dims: [0]
  augment_feature_repr: position
  augment_label: ''
  augment_label_dims: 0
  transform: none
  edge_encoder: True
  edge_encoder_name: roland
  edge_encoder_bn: True
  node_encoder: False
  node_encoder_name: roland
  node_encoder_bn: True
transaction:
  keep_ratio: linear
  snapshot: True
  snapshot_freq: M  # M=monthly, W=weekly, D=daily.
  check_snapshot: False
  history: rolling
  horizon: 1
  pred_mode: at
  loss: supervised
  feature_int_dim: 16  # number of categorical edge features.
  # number of unique values for each categorical edge feature, for example, 1017 means the first categorical
  # edge feature (which is PayerBank in BSI dataset) has 1017 unique values. This needs to be modified based on
  # the dataset.
  feature_edge_int_num: [1017, 1018, 33, 33, 13, 13, 23, 23, 86, 86, 5, 5, 9, 9, 1, 1]
  feature_node_int_num: [1, 1]
  feature_amount_dim: 16
  feature_time_dim: 16
train:
  batch_size: 32
  eval_period: 5
  ckpt_period: 400
  # See ./GraphGym_dev/graphgym/contrib/train/ for avaliable for `mode`.
  mode: new_hetero  # which training module to use.
model:
  # See ./GraphGym_dev/graphgym/contrib/network/ for all options.
  type: hetero_gnn_recurrent
  loss_fun: cross_entropy
  edge_decoding: concat  # Only use node embeddings.
  graph_pooling: add
gnn:
  layers_pre_mp: 2  # number of fully-connected before GNN.
  layers_mp: 2  # number of GNN layers.
  layers_post_mp: 2  # number of fully-connected after GNN.
  dim_inner: 128  # dimension of hidden layers in GNN.
  # See ./GraphGym_dev/graphgym/contrib/layer/ for all options.
  layer_type: generaledgeheteconv_complete
  stage_type: stack
  batchnorm: True
  act: prelu
  dropout: 0.0
  agg: add
  att_heads: 4
  normalize_adj: False
optim:
  optimizer: adam
  base_lr: 0.01
  max_epoch: 100
```
