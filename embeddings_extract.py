
# This code calculates the embeddings for the training and test set, once you've trained your model, and writes them to disk for further analysis.

# based on t-SNE.ipynb and visualise.py

import os
import numpy as np
import pandas as pd
from tools import utils
import torch
import torch.functional as F
from copy import deepcopy
from scipy.stats import special_ortho_group

scaler = torch.cuda.amp.GradScaler()

"""
numcl = 10
projmode = 'torusN'
"""

def l2norm(arr, keepdims=False):
    "Just to find the norm of an embedding"
    return np.sqrt(np.sum(arr ** 2, axis=1, keepdims=keepdims))

def torus_project_l2method(vals):
    "Applies L2-normalisation, but pairwise to dims rather than all dims at once, resulting in a Clifford torus projection"
    (batchsize, ndims) = np.shape(vals)
    circlesview = np.reshape(vals, newshape=(batchsize, ndims//2, 2))
    circlesview /= np.sqrt(np.sum(circlesview ** 2, axis=2, keepdims=True))
    return np.reshape(circlesview, newshape=(batchsize, ndims)) * np.sqrt(2.0/ndims)  # the final item here scales it all back to unit norm

def statstring(arr):
    amin = arr.min()
    amax = arr.max()
    pc05 = np.percentile(arr,  5)
    pc95 = np.percentile(arr, 95)
    return f"[{amin:.2f}, {pc05:.2f}, {pc95:.2f}, {amax:.2f}]"


#########################
os.makedirs(f"embeddings", exist_ok=True)

for numcl, projmodes, suffixes, projdims in [
        (10 , ["torusN"], ['_koleo0', '_koleo0-001', '_koleo0-1', '_koleo1'], [16]),
        #(10 , ["torusN", 'torusC', 'sphere'], ['_koleo0', '_koleo0-001', '_koleo0-1', '_koleo1'], [16, 32, 64, 128]),
        #(100, ["torusN", 'torusC', 'sphere'], ['_koleo0', '_koleo0-001', '_koleo0-1', '_koleo1'], [16, 32, 64, 128]),
        ]:

    for projection_dim in projdims:

        rotator = special_ortho_group(projection_dim, seed=54321).rvs()  # Random rotation (frozen) to inspect effect of rotation on outcome

        for projmode in projmodes:
            for suffix in suffixes:

                if projection_dim==32 and projmode in ["torus", "torusC"] and suffix=='_koleo0-001' and numcl==10:
                    continue  # skip a case which failed to train
                if projection_dim in [16, 32] and projmode in ["torus", "torusC"] and suffix=='_koleo1' and numcl==100:
                    continue  # skip a case which failed to train
                if projection_dim==16 and projmode in ["torus", "torusC"] and suffix in ['_koleo0-1', '_koleo1'] and numcl==10:
                    continue  # skip a case which failed to train
                #if projection_dim not in [32]:
                #    continue # HARD CODE to make it easy to manually skip some cases

                logging_name = f'cifar{numcl}_D{projection_dim}_{projmode}{suffix}'
                ckpt_pretrained = f'weights/supcon_stage_first_resnet18_{logging_name}/swa'
                logging_name += "_swa"

                print("=================================================")
                print(f"Loading model: {ckpt_pretrained}")
                data_dir = f'data/cifar{numcl}'
                num_classes = numcl
                batch_sizes = {
                        "train_batch_size": 20,
                            'valid_batch_size': 20
                            }
                num_workers = 16
                backbone = 'resnet18'
                stage = 'first'

                transforms = utils.build_transforms(second_stage=(stage == 'second'))
                loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == 'second'))

                ######################

                model = utils.build_model(backbone, second_stage=(stage == 'second'), num_classes=num_classes, ckpt_pretrained=ckpt_pretrained, projection_dim=projection_dim, projmode=projmode).cuda()
                model.eval()

                val_embeddings_orig, val_labels = utils.compute_embeddings(loaders['valid_loader'],          model, scaler, donormalise=False)
                trn_embeddings_orig, trn_labels = utils.compute_embeddings(loaders['train_features_loader'], model, scaler, donormalise=False)
                print(f"Shape of embeddings (trn): {np.shape(trn_embeddings_orig)}. Shape of labels: {np.shape(trn_labels)}")
                print(f"Shape of embeddings (val): {np.shape(val_embeddings_orig)}. Shape of labels: {np.shape(val_labels)}")

                for variant in ['', '_tra', '_rot']:
                    val_embeddings = deepcopy(val_embeddings_orig)
                    trn_embeddings = deepcopy(trn_embeddings_orig)

                    # Apply variants in the unconstrained space, before normalisation
                    if variant=='_tra':
                        val_embeddings += 0.5
                        trn_embeddings += 0.5
                    elif variant=='_rot':
                        val_embeddings = np.dot(val_embeddings, rotator)
                        trn_embeddings = np.dot(trn_embeddings, rotator)

                    # Apply normalisation
                    if projmode in ['torus', 'torusC']:
                        # The main code implements torus-wrapping outside of the network, via Clifford torus. Here we stay in the flat torus.
                        val_embeddings = np.mod(val_embeddings, 1.0)
                        trn_embeddings = np.mod(trn_embeddings, 1.0)
                    elif projmode in ['torul', 'torusN']:
                        val_embeddings = torus_project_l2method(val_embeddings)
                        trn_embeddings = torus_project_l2method(trn_embeddings)
                    elif projmode in ['hyprs', 'sphere']:
                        # L2-norm is indeed implemented in the main body, but we turned it off here (donormalise=False) so we can modify the values. So now we need to:
                        val_embeddings /= l2norm(val_embeddings, keepdims=True)
                        trn_embeddings /= l2norm(trn_embeddings, keepdims=True)
                    else:
                        raise ValueError(f"Unknown projmode: {projmode}")

                    print("   Mins: " + statstring(val_embeddings.min(axis=0)) + "   Means: " + statstring(val_embeddings.mean(axis=0)) + " Maxs: " + statstring(val_embeddings.max(axis=0)) + " Norms: " + statstring(l2norm(val_embeddings)) + "    " + variant)

                    pd.DataFrame(trn_embeddings).to_csv(f"embeddings/supcon_first_stage_{logging_name}{variant}_trn_embed.csv", index=False, header=False)
                    pd.DataFrame(val_embeddings).to_csv(f"embeddings/supcon_first_stage_{logging_name}{variant}_val_embed.csv", index=False, header=False)
                    pd.DataFrame(val_labels).to_csv(f"embeddings/supcon_first_stage_{logging_name}{variant}_val_labels.csv", index=False, header=False)
                    pd.DataFrame(trn_labels).to_csv(f"embeddings/supcon_first_stage_{logging_name}{variant}_trn_labels.csv", index=False, header=False)

