# Expressive-GNN

## Dependencies
numpy

scipy

pytorch

networkx

pandas

pickle

sklearn

## Run
for simple graph 

```
python graph_classification.py --dataset MUTAG --hidden_dim 16 --phi MLP --device 0 --fold_idx 0 --lr 0.01 --agg cat  --first_phi 
```

for attributed graph 
```
python graph_classification.py --dataset Synthie --hidden_dim 16 --phi MLP --device 0 --fold_idx 0 --lr 0.01 --agg cat --attribute --first_phi 
```


