#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import logging
import math
import os
import sys
import time
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def main(args):
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    embedding_layer = models[0].encoder
    embedding_weights = embedding_layer.embed_tokens.weight
    assert len(src_dict.symbols) == embedding_weights.shape[0]

    tokens = "love reading ! it ' s a means of sharing information and ideas reading is one of my favorite ways to spend my time . my favorite book series is harry potter by j . k . rowling . most . although children ' s literature also includes non - fiction , poetry and drama . i also enjoy fiction . what is your favorite children ' s fiction novel ? i would say harry potter is mine . k . rowling . many people love that series ! reading requires continuous practice development and refinement is reading a means of learning how to do something you ' ve never tried to do before ? reading is one of my favorite ways to spend my time . my favorite book series is harry potter by j . k . rowling . british author".split()
    tokens = list(set(tokens))
    print(len(tokens))
    vectors = []
    for token in tokens:
        index = src_dict.index(token)
        vectors.append(embedding_weights[index].tolist())

    km = KMeans(n_clusters=3)
    pca = PCA(n_components=2)
    vectors_ = pca.fit_transform(vectors)  # 降维到二维
    y_ = km.fit_predict(vectors_)  # 聚类
    colors = ["#abdda4", "#2b83ba", "#d7191c"]
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    plt.scatter(vectors_[:, 0], vectors_[:, 1], c=y_, cmap=clrmap)  # 将点画在图上
    keys = [ 'rowling', 'british', 'author', 'harry', 'potter']

    # keys = [ 'a', 'husky', 'is', 'polar', 'used', 'regions', 'sled', 'dog', 'could', 'would', 'need', 'do', 'long', 'popular', 'large', 'in', 'for', 'with']
    for key in keys:  # 给每个点进行标注
        i = tokens.index(key)
        offset = 0
        if key == 'harry':
            offset = 0.02
    # for i, token in enumerate(tokens):
        plt.annotate(s=tokens[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
                     xytext=(vectors_[:, 0][i], vectors_[:, 1][i]-offset),)
                     # arrowprops=dict(facecolor='black', width=0.01, headwidth=0.05, headlength=0.05))

    rowling = np.array([vectors_[:, 0][tokens.index('rowling')], vectors_[:, 1][tokens.index('rowling')]])
    british = np.array([vectors_[:, 0][tokens.index('british')], vectors_[:, 1][tokens.index('british')]])
    author = np.array([vectors_[:, 0][tokens.index('author')], vectors_[:, 1][tokens.index('author')]])
    print("distance between rowling and british:", np.linalg.norm(rowling-british))
    print("distance between rowling and author:", np.linalg.norm(rowling-author))
    # plt.annotate(str(round(np.linalg.norm(husky-polar), 3)),
    #              xy=(vectors_[:, 0][tokens.index('polar')], vectors_[:, 1][tokens.index('polar')]),
    #              textcoords=(vectors_[:, 0][tokens.index('husky')], vectors_[:, 1][tokens.index('husky')]+0.1),
    #              xytext=(vectors_[:, 0][tokens.index('husky')], vectors_[:, 1][tokens.index('husky')]),
    #              arrowprops=dict(facecolor='black', width=0.01, headwidth=0.05),
    #              )

    plt.show()
    # if 'our' in args.path:
    #     plt.savefig('our.pdf')
    # else:
    #     plt.savefig('base.pdf')
    plt.close()

def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    args.path = '../our.pt'
    distributed_utils.call_main(args, main)
    args.path = '../transformer.pt'
    distributed_utils.call_main(args, main)

if __name__ == "__main__":
    cli_main()
