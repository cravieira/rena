from functools import partial
from dotmap import DotMap
import numpy as np
from scipy.stats import norm
import torch
import torchhd
import torchhd.functional as functional
from torchhd.tensors.base import VSATensor
from torchhd.tensors.map import MAPTensor
from torchhd.tensors.fhrr import FHRRTensor
from torchhd.tensors.bsc import BSCTensor
from torchhd.tensors.cgr import CGRTensor
from torchhd.types import VSAOptions
import src.hw_rand as hw_rand
import src.tracer as tracer
from tqdm import tqdm

def MAP_weighted_bundling(codebook, weights):
    """Construct a MAP bundle considering an weigth vector

    The returned hypervector is a normalized MAP vector considering the weights.

    Args:
        codebook (VSATensor): The hypervector codebook
        weights (Tensor): An array of weights

    Shapes:
        - codebook: :math:`(*, n, d)`
        - weights: :math:`(*, n)`
        - Output: :math:`(*, d)`
    """
    output = functional.dot_similarity(weights, codebook.transpose(-2, -1)).squeeze(-2)
    normalized = output.sign()
    return normalized

def FHRR_weighted_bundling(codebook, weights):
    """Construct a FHRR bundle considering an weigth vector

    The returned hypervector is a normalized FHRR vector considering the weights.

    Args:
        codebook (VSATensor): The hypervector codebook
        weights (Tensor): An array of weights

    Shapes:
        - codebook: :math:`(*, n, d)`
        - weights: :math:`(*, n)`
        - Output: :math:`(*, d)`
    """
    output = torch.matmul(weights.to(torch.complex64), codebook).squeeze(-2)
    # Project phasors to unit length
    normalized = output.normalize()
    return normalized

def BSC_weighted_bundling(codebook, weights):
    """Construct a BSC bundle considering an weigth vector

    The returned hypervector is a normalized BSC vector considering the weights.

    Args:
        codebook (VSATensor): The hypervector codebook
        weights (Tensor): An array of weights produced by dot similarity in the range of [-D, D]

    Shapes:
        - codebook: :math:`(*, n, d)`
        - weights: :math:`(*, n)`
        - Output: :math:`(*, d)`
    """
    D = codebook.shape[-1]
    weights = weights.squeeze(-2) # Remove useless dimension inserted by dot_product. weights = [Batch, Factors, CB Size]
    B, F, M = weights.shape
    A = 2 # Number of accumulators per dimension. Considering binary values
    accs = torch.zeros((B, F, M, D, A))
    cb = codebook.reshape((1, F, M, D, 1)).expand((B, -1, -1, -1, -1)).to(torch.int64) # cb = [B, F, M, D, 1]
    acc_inc = weights.unsqueeze(-1).expand((-1, -1, -1, D)).unsqueeze(-1) # acc_inc = [B, F, M, D, 1]
    accs.scatter_(-1, cb, acc_inc)
    _, max_ind = accs.sum(-3).max(-1) # max_acc = [B, F]
    out = functional.ensure_vsa_tensor(max_ind, vsa='BSC', dtype=torch.bool)
    return out

def CGR_weighted_bundling(codebook, weights, /, cgr_bundle=None):
    """Construct a CGR bundle considering an weigth vector

    The returned hypervector is a normalized CGR vector considering the weights.

    Args:
        codebook (VSATensor): The hypervector codebook
        weights (Tensor): An array of weights produced by dot similarity in the range of [-D, D]

    Shapes:
        - codebook: :math:`(*, n, d)`
        - weights: :math:`(*, n)`
        - Output: :math:`(*, d)`
    """
    D = codebook.shape[-1]
    weights = weights.squeeze(-2) # Remove useless dimension inserted by dot_product. weights = [Batch, Factors, CB Size]
    B, F, M = weights.shape
    A = codebook.block_size # Number of accumulators per dimension

    if cgr_bundle == "mode":
        # CGR mode bundling
        # The result value is the most frequent one. Uses block_size accumulators
        # Find indices for accumulators
        cb = codebook.reshape((1, F, M, D)).expand((B, -1, -1, -1)) # cb = [B, F, M, D]
        opp_cb =  torch.remainder((cb+A//2), A) # opp_cb = [B, F, M, D]

        dot_sim = weights.unsqueeze(-1).expand(-1, -1, -1, D) # sim = [B, F, M, D]
        inc_ind = torch.where(dot_sim >= 0, cb, opp_cb) # inc_ind = [B, F, M, D]

        inc_ind = inc_ind.unsqueeze(-1) # [B, F, M, D, 1]
        inc = weights.abs().reshape(B, F, M, 1, 1).expand(-1, -1, -1, D, -1) # inc = [B, F, M, D, 1]

        accs = torch.zeros((B, F, M, D, A))
        accs.scatter_(-1, inc_ind, inc)
        _, max_ind = accs.sum(-3).max(-1) # max_acc = [B, F]

        out = functional.ensure_vsa_tensor(max_ind, vsa='CGR', dtype=codebook.dtype)
        out.block_size = codebook.block_size # Fix ensure_vsa_tensor not propperly setting the block_size
        return out
    elif cgr_bundle == "opposite":
        # CGR bundling with folded accumulators
        # Mimic the FHRR behavior by sharing the same accumulator for opposite indices.
        # Find the accumulator indices to be incremented
        inc_ind = torch.remainder(codebook, A//2).view((1, F, M, D, 1)).expand(B, -1, -1, -1, -1) # inc_ind = [B, F, M ,D, 1]

        # Find the incremented values.
        inv_val = codebook >= (A//2) # should we invert the weight values? inv_val = [F, M, D]
        inv_val = inv_val.unsqueeze(0).expand(B, -1, -1, -1)
        inc_val = weights # inc_val = [B, F, M]
        inc_val = inc_val.view((B, F, M, 1)).repeat(1, 1, 1, D)
        inc_val[inv_val] = -inc_val[inv_val]
        inc_val = inc_val.unsqueeze(-1) # inc_val = [B, F, M, D, 1]

        # Find indices for accumulators
        # By using codebook % (A//2), it is possible to assign opposite indices to the same accumulator
        inc_ind =  torch.remainder(codebook, A//2).view(1, F, M, D, 1).expand(B, -1, -1, -1, -1) # opp_cb = [B, F, M, D, 1]

        # Do accumulation
        accs = torch.zeros((B, F, M, D, A//2))
        accs.scatter_(-1, inc_ind, inc_val)
        acc_sum = accs.sum(-3) # acc_sum = [B, F, D, A]
        _, max_ind = acc_sum.abs().max(-1, keepdim=True) # max_ind = [B, F, D, 1]
        neg_chosen = acc_sum.gather(-1, max_ind) < 0
        max_ind[neg_chosen] = max_ind[neg_chosen] + (A//2)
        max_ind = max_ind.squeeze(-1) # Squeeze accumulator dim. max_ind = [B, F, D]

        out = functional.ensure_vsa_tensor(max_ind, vsa='CGR', dtype=codebook.dtype)
        out.block_size = codebook.block_size # Fix ensure_vsa_tensor not propperly setting the block_size
        return out
    else:
        raise RuntimeError(f"Invalid cgr_bundle parameter \"{cgr_bundle}\"")

# Attention computation #
# Compute the similarity of a query with vectors in a codebook
def _sim_normalize(x, dim):
    """
    Normalize a similarity tensor by the number of hypervector dimensions so
    that the similarity is in the range [-1, 1]
    """
    return x / dim

def MAP_attention(new_estimates, domains):
    dot_sim = functional.dot_similarity(new_estimates, domains)
    return _sim_normalize(dot_sim, domains.shape[-1])

def BSC_attention(new_estimates, domains):
    dot_sim = functional.dot_similarity(new_estimates, domains)
    return _sim_normalize(dot_sim, domains.shape[-1])

def FHRR_attention(new_estimates, domains):
    return functional.cosine_similarity(new_estimates, domains)

def CGR_attention(new_estimates, domains):
    dot_sim = functional.dot_similarity(new_estimates, domains)
    return _sim_normalize(dot_sim, domains.shape[-1])

# Activation functions #
# Activate on the output of an attention function
def calc_topaPT_threshold(topa, D, M):
    """
    Calculate topaPT ideal threshold.
    """
    std = np.sqrt(1/(4*D))*2
    percentage = 1-topa/M

    threshold = norm.ppf(percentage,0,std)
    threshold = np.round(threshold, 4)

    if threshold < 0 or np.isnan(threshold):
        threshold = 0

    return threshold

def topaPT(sim, threshold):
    """
    Top Attention Positive Threshold. Clear attention values below a threshold.

    This function is based on IBM's _topa_sparse_threshold_positiv() in their
    RN implementation. This function implements the activation function
    presented in the paper "In-memory factorization of holographic perceptual
    representations" and better discussed in the paper's companion
    sumplementary notes
    """
    # Old implementation - Slow
    #sim_t = sim.clone()
    #sim_t[sim_t < threshold] = 0

    ##if self._pullUp:
    ##    attn_values, attn_index = t.max(sim, dim=1)
    ##    attn_values_PU_mask = attn_values >= self._pullUp_thresh

    ##    # HIGH LEVEL CYCLES
    ##    sim[attn_values_PU_mask] = 0
    ##    sim[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]

    ## Commented out in IBM code since the noise injection after the projection step can naturally recover from an all 0s topaPT activation
    #sum = torch.sum(sim_t,-1)
    ##if (sum == 0).any():
    ##    print(f"TopaPT safety trigger. {(sum == 0).sum()}")
    #sim_t[sum == 0] = sim[sum == 0] # if all zero recover by not applying any theshold
    #return sim_t

    # New implementation - Faster
    sim[sim < threshold] = 0

    #if self._pullUp:
    #    attn_values, attn_index = t.max(sim, dim=1)
    #    attn_values_PU_mask = attn_values >= self._pullUp_thresh

    #    # HIGH LEVEL CYCLES
    #    sim[attn_values_PU_mask] = 0
    #    sim[attn_values_PU_mask, attn_index[attn_values_PU_mask]] = attn_values[attn_values_PU_mask]

    # Commented out in IBM code since the noise injection after the projection step can naturally recover from an all 0s topaPT activation
    #sum = torch.sum(sim,-1)
    ##if (sum == 0).any():
    ##    print(f"TopaPT safety trigger. {(sum == 0).sum()}")
    #sim_t[sum == 0] = sim[sum == 0] # if all zero recover by not applying any theshold

    return sim

def topaPT_safe(sim, threshold):
    """
    Top Attention Positive Threshold Safe. Clear attention values below a
    threshold or do nothing in case all similarities were reseted.

    This function is based on IBM's _topa_sparse_threshold_positiv() in their
    RN implementation. This function implements the activation function
    presented in the paper "In-memory factorization of holographic perceptual
    representations" and better discussed in the paper's companion
    sumplementary notes
    """
    sim_t = sim.clone()
    sim_t = topaPT(sim_t, threshold)

    sum = torch.sum(sim_t,-1)
    if (sum == 0).any():
        print(f"TopaPT safety trigger. {(sum == 0).sum()}")
    sim_t[sum == 0] = sim[sum == 0] # if all zero recover by not applying any theshold

    return sim_t


VECTOR_PROJECTIONS = {
        "MAP": MAP_weighted_bundling,
        "BSC": BSC_weighted_bundling,
        "FHRR": FHRR_weighted_bundling,
        "CGR": CGR_weighted_bundling,
        }

ATTENTION_FUNCTIONS = {
        "MAP": MAP_attention,
        "BSC": BSC_attention,
        "FHRR": FHRR_attention,
        "CGR": CGR_attention,
        }

ACTIVATION_FUNCTIONS = {
        "identity": torch.nn.Identity(),
        "topaPT": topaPT,
        "topaPT-safe": topaPT_safe,
        }

def normal_noise(sim, std=1):
    """docstring for normal_noise"""
    noise = torch.normal(0,std, sim.shape)
    # Bug: torch.normal is creating tensors on CPU and ignoring
    # torch.default_device().
    noise = noise.to(sim.device)
    noisy_sim = sim + noise
    return noisy_sim

def _xorshift_noise(sim, max, dim, t):
    """Common xorshift noise implementation"""
    noise = hw_rand.apply_range(t, max)
    norm_noise = noise/dim
    noisy_sim = sim + norm_noise
    return noisy_sim

def xorshift_noise(sim, prng, max, dim):
    """docstring for xorshift_noise"""
    t = hw_rand.tensor1D(prng, sim.shape)
    noise = hw_rand.apply_range(t, max)
    norm_noise = noise/dim
    noisy_sim = sim + norm_noise
    return noisy_sim

def parallel_np_xorshift_noise(sim, prng, max, dim):
    """docstring for xorshift_noise"""
    #t = hw_rand.tensor1D(prng, sim.shape)
    t = prng.next()
    t = t.to(sim.device) # t is created from numpy. Move it to sim's device
    t = t[0: sim.shape.numel()] # The number of batches decreases along RN execution
    t = t.view(sim.shape)
    noisy_sim = _xorshift_noise(sim, max, dim, t)
    return noisy_sim

NOISE_FUNCTIONS = {
        "identity": torch.nn.Identity(),
        "normal": normal_noise,
        "xorshift": xorshift_noise,
        "parallel_np_xorshift": parallel_np_xorshift_noise,
        }

def clone_not_converged(input, not_converged):
    """docstring for clone_not_converged"""
    #input_f = input[not_converged].clone()
    input_f = input[not_converged]
    return input_f

def rn_unbind_index(output, not_converged, mask):
    o_not_converged = output[not_converged]
    return o_not_converged[:, mask]

def rn_unbind(F: int, current_feature: int, decoding: str, input_f, output, inv_estimates, not_converged):
    """docstring for rn_unbind"""
    # Create indexing mask
    mask = torch.arange(F)
    index_to_exclude = current_feature
    mask = torch.cat([mask[:index_to_exclude], mask[index_to_exclude+1:]])

    if decoding == 'sequential':
        #o_not_converged = output[not_converged]
        #inv = functional.inverse(o_not_converged[:, mask]).multibind()
        inv = functional.inverse(rn_unbind_index(output, not_converged, mask)).multibind()

    # TODO: implement support for parallel decoding
    # TODO: add support for RN tracing as the traditional loop does

    input_f = functional.bind(input_f, inv)

    return input_f

    # Traditional approach with python loop - Too slow
    #for f_dec in range(F):
    #    if current_feature != f_dec:
    #        if decoding == "sequential":
    #            inv = functional.inverse(output[not_converged, f_dec])
    #            input_f = functional.bind(input_f, inv)

    #            #tracer_unbind_features.append(inv)
    #        else:
    #            input_f = functional.bind(input_f, inv_estimates[not_converged, f_dec])

    #            #tracer_unbind_features.append(inv_estimates[f_dec])
    #return input_f

def convergence_detection(sim_act_noisy, state_converged, not_converged, convergence_threshold, f):
    """docstring for convergence_detection"""
    sim_act_noisy = sim_act_noisy.squeeze(1).squeeze(1)
    state_converged[not_converged, f] = (torch.max(sim_act_noisy, dim=1)[0] > convergence_threshold)
    return state_converged

def resonator(
        input: VSATensor,
        estimates: VSATensor,
        domains: VSATensor,
        state_converged,
        convergence_idx,
        convergence_iter,
        args,
        vsa="MAP",
        decoding="parallel",
        convergence_threshold=0.625,
        activation=None,
        noise=None,
    ) -> VSATensor:
    """A step of the resonator network that factorizes the input.

    Given current estimates for each factor, it returns the next estimates for those factors.

    Args:
        input (VSATensor): The hypervector to be factorized.
        estimates (VSATensor): The current estimates of the factors, typically starts as a multiset of the domain.
        domains (VSATensor): The domains of each factor containing all possible factors.

    Shapes:
        - Input: :math:`(*, d)`
        - Estimates: :math:`(*, n, d)`
        - Domains: :math:`(*, n, m, d)`
        - Output: :math:`(*, n, d)`
    """
    input = functional.ensure_vsa_tensor(input)
    estimates = functional.ensure_vsa_tensor(estimates)
    domains = functional.ensure_vsa_tensor(domains)

    tensor_types = {
            "MAP": MAPTensor,
            "BSC": BSCTensor,
            "FHRR": FHRRTensor,
            "CGR": CGRTensor
    }
    t_type = tensor_types[vsa]

    if not isinstance(input, t_type):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {input.__class__.__name__}"
        )

    if not isinstance(estimates, t_type):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {estimates.__class__.__name__}"
        )

    if not isinstance(domains, t_type):
        raise ValueError(
            f"Resonator currently only supports Multiply-Add-Permute (MAPTensor) VSA model, provided: {domains.__class__.__name__}"
        )

    f_attention = ATTENTION_FUNCTIONS[vsa]
    f_projection = VECTOR_PROJECTIONS[vsa]
    if vsa == "CGR":
        f_projection = partial(f_projection, cgr_bundle=args.cgr_bundle)

    f_activation = activation
    if activation is None:
        f_activation = ACTIVATION_FUNCTIONS["identity"]

    f_noise = noise
    if noise is None:
        f_noise = NOISE_FUNCTIONS["identity"]

    features = estimates.size(-2) # Get the number of features
    D = estimates.size(-1) # Get the number of dimensions

    if decoding == "sequential":
        #output = estimates.clone() # TODO: maybe optimize this by removing cloning
        output = estimates # TODO: maybe optimize this by removing cloning
    else:
        inv_estimates = torchhd.inverse(estimates)
        output = estimates.clone() # TODO: maybe this could become torch.zeros to create a placeholder for output

    # Alternative 1 - Boolean Mask indexing - Slow
    # not_converged = ~convergence_idx

    # Alternative 2 - Convert boolean to integer Mask indexing - Fast
    # TODO: Maybe remove boolean indexing from the parent rn_top function too
    not_converged = torch.arange(convergence_idx.shape[0])[~convergence_idx]

    for f in range(features):
        tracer.set_feature(f)

        # Prepare input - TODO: Try to remove clone
        input_f = clone_not_converged(input, not_converged)
        #input_f = input[not_converged].clone()

        tracer_unbind_features = []

        ## Unbind
        #for f_dec in range(features):
        #    if f != f_dec:
        #        if decoding == "sequential":
        #            inv = functional.inverse(output[not_converged, f_dec])
        #            input_f = functional.bind(input_f, inv)

        #            tracer_unbind_features.append(inv)
        #        else:
        #            input_f = functional.bind(input_f, inv_estimates[not_converged, f_dec])

        #            tracer_unbind_features.append(inv_estimates[f_dec])

        #new_estimates = input_f

        new_estimates = rn_unbind(features, f, decoding, input_f, output, None, not_converged)

        # Tracer to debug RN computation
        f_d = {f"f_{i}": tracer_unbind_features[i] for i in range(len(tracer_unbind_features))}
        tracer.register_op({
            "name": "unbind",
            "input": input,
            "output": new_estimates
            } | f_d
        )

        # TODO: Analyze dimensions. Maybe some squeeze/unsqueeze operations could be removed
        # Obtain similarity of the unbound estimates and each codebook.
        # Similarities are in [-1, 1].
        similarity = f_attention(new_estimates.unsqueeze(-2), domains[f])

        tracer.register_op({
            "name": "similarity",
            "sim": (similarity[0]*D).int()
            })

        # Apply noise before attention
        sim_noisy = f_noise(similarity)

        # Apply activation function
        sim_act = f_activation(sim_noisy)

        # Apply noise before projection
        #sim_act_noisy = f_noise(sim_act)
        sim_act_noisy = sim_act

        # Construct new prediction vectors with weighted bundling
        # Add a placeholder dimension for the features to call weighted_bundling. sim_act_noisy = [B, 1, 1, M])
        sim_act_noisy = sim_act_noisy.unsqueeze(1)
        output[not_converged, f] = f_projection(domains[f].unsqueeze(0), sim_act_noisy).squeeze(-2)

        # Has this feature converged in this iteration?
        # Supress placeholder dimensions. [B, 1, 1, M] -> [B, M]
        #sim_act_noisy = sim_act_noisy.squeeze(1).squeeze(1)
        #state_converged[not_converged, f] = (torch.max(sim_act_noisy, dim=1)[0] > convergence_threshold)

        state_converged = convergence_detection(sim_act_noisy, state_converged, not_converged, convergence_threshold, f)

    return output, state_converged

def predict_estimates(estimates, codebooks):
    """
    Make the final RN prediction based on the estimates and the codebooks

    :param estimates [torch.Tensor]: A tensor of estimates in the form [B, F, D]
    :param codebooks [torch.Tensor]: A tensor of codebooks in the form [F, M, D]
    """
    """docstring for predict_estimates"""
    # Implementation 1 - matmul based
    # The code below is ommitted due to potential high RAM usage if M is too
    # large. The returned 'sim' tensor has shape [B, F, M].
    ## Make predictions using absolute similarity. Some RN predictions may
    ## converge to maximal disimilarity. This issue has been reported in
    ## other RN implementations as a "degeneracy". Example of degeneracy
    ## in other implementations:
    ## https://github.com/spencerkent/resonator-networks/blob/179da7e060b91de7c29d130e8941b3e03c295499/resonator_networks/dynamics/rn_pytorch.py#L156C1-L159C69
    ## https://github.com/IBM/in-memory-factorizer/blob/a353f1e918dcb515cad4a89c8e47ce24668954a7/models/densebipolarbatched.py#L656
    #sim = torchhd.cosine_similarity(estimates.unsqueeze(-2), codebooks).squeeze(-2).abs()
    #final_estimates = sim.argmax(dim=-1)

    # Implementation 2 - Iterate over the batches
    B, F, _ = estimates.shape
    final_estimates = torch.zeros((B, F))
    est = estimates.unsqueeze(-2)
    for b in range(B):
        final_estimates[b] = torchhd.cosine_similarity(est[b], codebooks).squeeze(-2).abs().argmax(-1)

    return final_estimates

def rn_top(
        s,
        estimates,
        codebooks,
        max_iter,
        args,
        vsa,
        device=None,
        #f_act=ACTIVATION_FUNCTIONS["identity"],
        #f_noise=NOISE_FUNCTIONS["identity"],
        **kwargs
        ):
    """docstring for rn_top"""

    B, F, D = estimates.shape
    # Create convergence detection registers
    # Tracks which features have converged
    if device is None:
        device=torch.get_default_device()
    state_converged = torch.Tensor(B, F).zero_().type(torch.bool).to(device) # [B, F] <= False
    # Tracks which batches have converged
    convergence_idx = torch.Tensor(B).zero_().type(torch.bool).to(device) # [B] <= False
    # Tracks which iteration convergence occurs
    convergence_iter = torch.Tensor(B, 1).fill_(max_iter-1).to(device) # [B, 1] <= max_iter-1

    # TODO: Refactor how not converged batches are passed. Index the necessary work to do in the top function and pass only the work tbd.
    # Checking if this speeds up as intended might require observing the whole program execution
    for iter in tqdm(range(1, max_iter)):
        tracer.set_iter(iter)

        estimates, state_converged = resonator(s, estimates, codebooks, state_converged, convergence_idx, convergence_iter, args, vsa=vsa, **kwargs)

        # Update the array of converged batches
        convergence_idx = (torch.sum(state_converged, dim=1) > 0).type(torch.bool)
        # Write the iteration the convergence occurred
        # TODO: Maybe using a special viewed RO tensor istead of ones_like is faster
        convergence_iter[convergence_idx] = torch.min(convergence_iter[convergence_idx], torch.ones_like(convergence_iter[convergence_idx])*iter)

        # Finish RN earlier if all batches have converged
        if convergence_idx.all():
            break

    # Make final prediction
    final_estimates = predict_estimates(estimates, codebooks)

    return final_estimates, convergence_iter
