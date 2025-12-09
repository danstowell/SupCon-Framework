import numpy as np
import pandas as pd
from tools import utils
import faiss
import scipy.stats

#############################
# Evaluate embedding performance, using different quantisation modes.
# QUANT MODES:
# * unquant
# * 8bit-per-dim quant -- simply quantise uniform grid across the known range
# * 1bit-per-dim quant -- similar. for a full hypersphere this is sign-quantisation.
# * PQ -- since the hypersph is not expected to be good under grid quant, we need to provide this to give hypersph a chance to quantise well.
##############################
"""
numcl = 100
projmode = 'torul'
projection_dim = 16
koleo = "0-1"
suffix = ''
quantmode = "pq8_1"
pq_nbits, pq_m = (8, 1)
"""

outpath = "tbl_embeddings_quant_eval.csv"
outfp = False

def gridquant_calc_params(df, nbit):
    "grid quantisation for every feature dimension independently."
    nvals = 2 ** nbit
    subber = df.min(axis=0)
    scaler = nvals / (df - subber).max(axis=0)
    if np.isinf(scaler).any():
        raise ValueError("Collapsed range")
    return (nvals, subber, scaler)

def gridquant_apply(df, params):
    "returns floating-point data but quantised to an integer grid, and then scaled to [0,1]"
    (nvals, subber, scaler) = params
    dfq = np.minimum(np.floor((df - subber) * scaler), nvals-1) / nvals
    return dfq

def shannon_entropy(df, totnbits):
    "Estimates the discrete (Shannon) entropy based on a histogram of the discrete vector values"
    _, counts = np.unique(df, axis=0, return_counts=True)
    nitems = df.shape[0]
    # we assume for simplicity (and tractable calculations) that the total num items LIMITS the total possible values,
    #  even though for high-bitrate the total possible values would be astronomical.
    if totnbits is None:
        possvals = nitems
    else:
        possvals = min(nitems, 2 ** totnbits)
    if len(counts) < possvals:  # ...thus, if-the-set-of-possible-values-hasnt-been-fully-explored
        counts = np.concatenate((counts, [1e-6] * (possvals-len(counts))))
    return scipy.stats.entropy(counts)

def decode_koleo_val(astr):
    "Supply the filename-encoded koleo weight, and this will return it as a number"
    if "e" not in astr:
        astr = astr.replace("-", ".")
    return np.array(astr, dtype=float)

for numcl, projmodes, koleos, suffixes, projdims in [
        (10 , ["torusN"], ['0', '0-001', '0-1', '1'], [''], [16]),
        #(10 , ["torusN", 'torusC', 'sphere'], ['0', '0-001', '0-1', '1'], [''], [16, 32, 64, 128]),
        #(100, ["torusN", 'torusC', 'sphere'], ['0', '0-001', '0-1', '1'], [''], [16, 32, 64, 128]),
       ]:

    for projection_dim in projdims:
        for projmode_raw in projmodes:
            for koleo in koleos:
                koleonum = decode_koleo_val(koleo)
                for suffix in suffixes:
                    
                    if projection_dim==32 and projmode_raw in ["torus", "torusC"] and koleo=='0-001' and numcl==10:
                        continue  # skip a case which failed to train
                    if projection_dim in [16, 32] and projmode_raw in ["torus", "torusC"] and koleo=='1' and numcl==100:
                        continue  # skip a case which failed to train
                    if projection_dim==16 and projmode_raw in ["torus", "torusC"] and koleo in ['0-1', '1'] and numcl==10:
                        continue  # skip a case which failed to train
                    #if projection_dim not in [2]:
                    #    continue # HARD CODE to make it easy to manually skip some cases

                    # Load the embeddings
                    logging_name = f'cifar{numcl}_D{projection_dim}_{projmode_raw}_koleo{koleo}_swa{suffix}'
                    trne = pd.read_csv(f'embeddings/supcon_first_stage_{logging_name}_trn_embed.csv', header=None)
                    vale = pd.read_csv(f'embeddings/supcon_first_stage_{logging_name}_val_embed.csv', header=None)
                    trnlbls = pd.read_csv(f'embeddings/supcon_first_stage_{logging_name}_trn_labels.csv', header=None).to_numpy().squeeze()
                    vallbls = pd.read_csv(f'embeddings/supcon_first_stage_{logging_name}_val_labels.csv', header=None).to_numpy().squeeze()

                    # For the CSV and plots we want to ensure we're using the "readable" keywords
                    logging_name = utils.standardise_projmode_name_in_loggingname(logging_name)
                    projmode = utils.standardise_projmode_name(projmode_raw)

                    # Convert the data to the format expected by torch and thus faiss
                    trne_np = np.ascontiguousarray(trne.to_numpy(), dtype=np.float32)
                    vale_np = np.ascontiguousarray(vale.to_numpy(), dtype=np.float32)

                    print(trne.shape)
                    print(trnlbls.shape)
                    print(vale.shape)
                    print(vallbls.shape)

                    # the "torusN" version is trained in clifford space. the optimal quant for this should be to unwrap it into flat torus, and thence treat it like torus.
                    if projmode=='torusN':
                        trne = utils.unwrap_pairwise_torus(trne)
                        vale = utils.unwrap_pairwise_torus(vale)
                        trne_np = utils.unwrap_pairwise_torus(trne_np, pandas=False)
                        vale_np = utils.unwrap_pairwise_torus(vale_np, pandas=False)

                    quantparamscached = {}

                    ###########################################################
                    # NB always run "8bit" before PQ, since PQ uses it as input
                    for quantmode in ['none', '8bit', '1bit',
                            'pq8_2', 'pq8_1',
                            'pq4_4', 'pq4_2']:
                        print("===========================================")
                        print(f"{logging_name}: quantmode {quantmode}")

                        if quantmode =='none': # no quant
                            dfqt = trne_np
                            dfqv = vale_np
                            totnbits = None # total number of bits used to encode one value - but we skip it here
                        elif quantmode.endswith("bit"): # n-bit grid
                            nbit = int(quantmode[:-3])

                            trne_np_TMP = trne_np
                            vale_np_TMP = vale_np
                            if nbit==1 and projmode=='torusN':
                                # Since 1bit is creating a pure Hamming space rather than a true "grid",
                                #  we will in fact stick with the Clifford version, giving same dims (same totnbits) as other projmodes
                                trne_np_TMP = utils.circular_torus_embed(trne_np_TMP)
                                vale_np_TMP = utils.circular_torus_embed(vale_np_TMP)

                            totnbits = nbit * trne_np_TMP.shape[1]  # total number of bits used to encode one value

                            try:
                                quantparams = gridquant_calc_params(trne_np_TMP, nbit)
                            except ValueError:
                                print("  WARNING: skipping {quantmode} due to collapsed range")
                                continue
                            dfqt = gridquant_apply(trne_np_TMP, quantparams)
                            dfqv = gridquant_apply(vale_np_TMP, quantparams)

                            if nbit==8:
                                # We're cacheing the data here so that we can re-use the quantised data within a different type of quantisation
                                # We're also explicitly coding to integer rather than [0,1]
                                quantparamscached[quantmode] = {
                                        'params': quantparams,  # NB NEED TO SCALE BACK UP
                                        'trne': (dfqt * quantparams[0]).astype(int),
                                        'vale': (dfqv * quantparams[0]).astype(int),
                                        }

                        elif quantmode.startswith("pq"):
                            pq_nbits, pq_m = map(int, quantmode[2:].split('_'))
                            pq_D = trne_np.shape[1]
                            if pq_D % pq_m != 0:
                                print(f"  WARNING: cannot apply {quantmode} to embeddings of dimension {pq_D}")
                                continue
                            pq_index = faiss.IndexPQ(pq_D, pq_m, pq_nbits)
                            print("   Training PQ...")
                            pq_index.train(quantparamscached['8bit']['trne'].astype('float32'))  # trne_np
                            print("   ...trained PQ.")
                            pq_index.add(quantparamscached['8bit']['trne'].astype('float32'))  # trne_np

                            # We quantise by training the PQ using trn (above); then at test time, finding the nearest trn-vector for each val-vector
                            pq_indices_v = pq_index.search(quantparamscached['8bit']['vale'].astype('float32'), 1)[1].flatten()  # vale_np
                            dfqv = np.array([pq_index.reconstruct(int(idx)) for idx in pq_indices_v])
                            pq_indices_t = pq_index.search(quantparamscached['8bit']['trne'].astype('float32'), 1)[1].flatten()  # tnre_np
                            dfqt = np.array([pq_index.reconstruct(int(idx)) for idx in pq_indices_t])
                            totnbits = pq_nbits * pq_m # total number of bits used to encode one value
                            del pq_index

                        ############ matches utils.validation_constructive():
                        calculator = utils.AccuracyCalculator(k=1)
                        query_embeddings     = np.ascontiguousarray(dfqv)
                        query_labels      = np.ascontiguousarray(vallbls)
                        reference_embeddings = np.ascontiguousarray(dfqt)
                        reference_labels  = np.ascontiguousarray(trnlbls)

                        # The primary evaluation will be the standard one conducted here.
                        # Note: in principle we don't need this clifford projection for inference
                        #   BUT we use it here as an easy way to include the wraparound behaviour
                        #   with ordinary distance functions, for our evaluation.
                        # If we unwrap torusN, we would also rewrap it here.
                        if projmode in ['torusC', 'torusN']:
                            query_embeddings = utils.circular_torus_embed(query_embeddings)
                            reference_embeddings = utils.circular_torus_embed(reference_embeddings)

                        all_embeddings = np.concatenate((query_embeddings, reference_embeddings), axis=0)

                        # Calculate some evaluation measures based on the distributions of the (quantised) data
                        nunique_val = np.unique(all_embeddings, axis=0).shape[0]
                        entropy = shannon_entropy(all_embeddings, totnbits)
                        # Circular variance. Given embeddings of unit L2, is 1 - l2mag(mean(vecs))
                        normalised_for_circvar = all_embeddings
                        normalised_for_circvar = normalised_for_circvar / np.sqrt(np.sum(normalised_for_circvar ** 2, axis=1, keepdims=True))
                        circmeanvec = np.mean(normalised_for_circvar, axis=0)
                        circvar = 1 - np.sqrt(np.sum(circmeanvec ** 2))
                        # calc the intrinsic dimensionality, just to report it
                        if projmode=='torusC':
                            projdimsintr = projection_dim
                        elif projmode=='torusN':
                            projdimsintr = projection_dim // 2
                        elif projmode=='sphere':
                            projdimsintr = projection_dim - 1

                        # Calculate some evaluation measures based on classification
                        acc_dict = calculator.get_accuracy(
                            query_embeddings,
                            query_labels,
                            reference_embeddings,
                            reference_labels,
                            ref_includes_query=False
                        )

                        del query_embeddings, query_labels, reference_embeddings, reference_labels, all_embeddings, normalised_for_circvar, calculator

                        print(acc_dict)

                        if not outfp:
                            outfp = open(outpath, "w")
                            outfp.write("dataset,numcl,projmode,projection_dim,dim_intr,koleo,suffix,quantmode,lbl,nunique_val,circvar,entropy," + ",".join(sorted(acc_dict.keys())) + "\n")
                        
                        outfp.write(f"{logging_name.split('_')[0]},{numcl},{projmode},{projection_dim},{projdimsintr},{koleonum},{suffix},{quantmode},{projmode}_{projection_dim}_koleo{koleo}{suffix}_{quantmode},{nunique_val},{circvar},{entropy}," + ",".join([str(v) for k,v in sorted(acc_dict.items())]) + "\n")

if outfp:
    outfp.close()
