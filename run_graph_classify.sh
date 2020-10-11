
# for simple graph
# dataset=MUTAG
phi=vdmd
for dataset in MUTAG PTC NCI1 PROTEINS 
do
for fold in {0..9}
do 
    python3 graph_classification.py --dataset $dataset --hidden_dim 16 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat   &
    python3 graph_classification.py --dataset $dataset --hidden_dim 32 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat   &
    python3 graph_classification.py --dataset $dataset --hidden_dim 64 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat    &
done
done


for dataset in COLLAB IMDBBINARY IMDBMULTI REDDITBINARY 
do
for fold in {0..9}
do 
    python3 graph_classification.py --dataset $dataset --hidden_dim 16 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat    &
    python3 graph_classification.py --dataset $dataset --hidden_dim 32 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat    &
    python3 graph_classification.py --dataset $dataset --hidden_dim 64 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat    &
done
done

# for attributed graph
# dataset=Synthie
# for fold in {0..9}
# do 
#     python3 graph_classification.py --dataset $dataset --hidden_dim 16 --phi $phi --device 7 --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi   &
#     python3 graph_classification.py --dataset $dataset --hidden_dim 32 --phi $phi --device 8 --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi  &
#     python3 graph_classification.py --dataset $dataset --hidden_dim 64 --phi $phi --device 9 --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi  &
#     wait
# done

# for dataset in ENZYMES FRANKENSTEIN PROTEINSatt SYNTHETICnew
# do
# for fold in {0..9}
# do 
#     python3 graph_classification.py --dataset $dataset --hidden_dim 16 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi   &
#     python3 graph_classification.py --dataset $dataset --hidden_dim 32 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi  &
#     python3 graph_classification.py --dataset $dataset --hidden_dim 64 --phi $phi --device $fold --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi  &
# done
# done

