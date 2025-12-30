r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=128,
        seq_len=128,
        h_dim=512,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.0003,
        lr_sched_factor=0.7,
        lr_sched_patience=1,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences instead of training on the whole text mainly for memory reasons, training on the entire corpus requires a lot of memory, as the length of the sequence increases, the consumption increases, and therefore splitting into short sequences allows for efficient training within memory limitations.
Training in smaller parts addresses the problems of exploding and vanishing gradients that we saw in the past by keeping them within a range of values ​​that allows for effective updating of the weights.
In addition, by dividing into mini-batches we allow the model to perform frequent updates after each batch, thus enabling convergence to an optimal solution more accurately and faster than by performing a single update at the end of the corpus.

"""

part1_q2 = r"""
It possible that the generated text shows memory longer than the sequence length Because Instead of reinitializing the model for each sequence, we use a hidden state from the end of one sequence as the initial state for the next one. 
This allows the hidden state to act as persistent memory, the model learns over time and well beyond the length of a single training sequence.

"""

part1_q3 = r"""
We not shuffling the order of batches when training Because, as we explained in the previous section, the hidden state moves from the end of a sequence to the beginning of the next sequence, and therefore continuity of information is required. If we shuffle the data, the model will not be able to learn the connections between the parts of the sentence,
and when we move the hidden state, the memory it holds will not teach our model with the level of accuracy and predictions we expected.

"""

part1_q4 = r"""
1. We lower the temperature for sampling Because It makes the model more conservative. By dividing the logits by a number less than 1, we sharpen the gaps between the logit values ​​and therefore also sharpen the probability distribution during the Softmax. 
The model gives higher weight to its safe option, thus reducing randomness and increasing the chance of getting a more accurate and consistent answer.

2. When the temperature is very high Because The model becomes more adventurous and less stable. Dividing the logit by a larger number flattens the probability distribution so that the gaps between the words are reduced, as is their softmax difference, and the model will ignore its preferences. Therefore, with a uniform probability distribution,
 we would expect more random and less accurate results, since the words get almost an equal chance of being selected, making it more uniform.And a less logically sensible output is expected.

3. When the temperature is very low Because The model becomes more repetitive and deterministic. Dividing the logits by a very small value increases the gaps and sharpens the probability distribution during their Softmax step, forcing the model to choose its safest option (the one with the highest score) and ignore the others. 
We expect a more logical and accurate output, but also repetitive and uncreative.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
