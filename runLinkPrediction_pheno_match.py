import argparse
import logging
import os
import random
import re

import numpy
import yaml
from tqdm import tqdm
from xn2v import CSFGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v.word2vec import ContinuousBagOfWordsWord2Vec
from xn2v import LinkPrediction
import xn2v
import sys

# Adapted from Vida's code to predict PPI links

regex = re.compile("<(.*[_|:])(.*)>")


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run link Prediction.")

    parser.add_argument('--make_edge_files', dest='make_edge_files', action='store_true')
    parser.add_argument('--dont_make_edge_files', dest='make_edge_files', action='store_false')
    parser.set_defaults(make_edge_files=True)

    parser.add_argument('--upheno_graph', type=argparse.FileType('r'),
                        help='Path to file with all edges in upheno graph')

    parser.add_argument('--equivalent_phenotypes', type=argparse.FileType('r'),
                        help='Path to file with edges/weights for equivalent phenotypes')

    parser.add_argument('--weight_multiplier', type=int, default=1,
                        help='Factor to multiply weight of phenotype edges to bias random walk')

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('--edge_embed_method', nargs='?', default='hadamard',
                        help='Embeddings embedding method of the positive training graph. '
                             'It can be hadamard, weightedL1, weightedL2 or average')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=80,
                        help='Number of walks per source. Default is 80.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--gamma', type=float, default=1,
                        help='hyperparameter for jumping from one network to another network '
                             'in heterogeneous graphs. Default is 1.')

    parser.add_argument('--useGamma', dest='useGamma', action='store_true',
                        help="True if the graph is heterogeneous, "
                             "False if the graph is homogeneous.")
    parser.add_argument('--dontUseGamma', dest='useGamma', action='store_false',
                        help="True if the graph is heterogeneous, "
                             "False if the graph is homogeneous.")
    parser.set_defaults(useGamma=True)
    parser.add_argument('--classifier', nargs='?', default='RF',
                        help="Binary classifier for link prediction, it should be either LR, RF or SVM")

    parser.add_argument('--type', nargs='?', default='heterogen',
                        help="Type of graph which is either homogen for homogeneous graph or heterogen for heterogeneous graph")

    parser.add_argument('--w2v-model', nargs='?', default='Skipgram',
                        help="word2vec model. It can be either Skipgram or CBOW")

    return parser.parse_args()


def learn_embeddings(walks, pos_train_graph, w2v_model):
    '''
    Learn embeddings by optimizing the Skipgram or CBOW objective using SGD.
    '''

    worddictionary = pos_train_graph.get_node_to_index_map()
    reverse_worddictionary = pos_train_graph.get_index_to_node_map()

    if w2v_model == "Skipgram":
        model = SkipGramWord2Vec(walks, worddictionary=worddictionary,
                                 reverse_worddictionary=reverse_worddictionary,
                                 num_steps=100)
    elif w2v_model == "CBOW":
        model = ContinuousBagOfWordsWord2Vec(walks, worddictionary=worddictionary,
                                             reverse_worddictionary=reverse_worddictionary,
                                             num_steps=100)
    else:
        print("[ERROR] enter Skipgram or CBOW")
        sys.exit(1)

    model.train(display_step=2)

    model.write_embeddings(args.embed_graph)


def linkpred(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph):
    """
    :param pos_train_graph: positive training graph
    :param pos_test_graph: positive test graph
    :param neg_train_graph: negative training graph
    :param neg_test_graph: negative test graph
    :return: Metrics of logistic regression as the results of link prediction
    """
    lp = LinkPrediction(pos_train_graph, pos_test_graph, neg_train_graph,
                        neg_test_graph,
                        args.embed_graph, args.edge_embed_method, args.classifier,
                        args.type)

    lp.prepare_lables_test_training()
    lp.predict_links()
    lp.output_classifier_results()
    lp.output_edge_node_information()


def make_iri_to_curie_map() -> dict:
    with open(os.path.join('data', 'curie_map.yaml')) as yaml_file:
        curie_map = yaml.safe_load(yaml_file)
    return {v: k for k, v in curie_map.items()}


def curieize(item, curie_map):
    curie_prefix = regex.sub(r'\1', item)
    id = regex.sub(r'\2', item)

    if curie_prefix in curie_map:
        return curie_map[curie_prefix] + ":" + id
    else:
        return item


def make_train_test_files(upheno_graph,
                          equiv_phenotypes,
                          weight_multiplier: int,
                          pos_train: str,
                          pos_test: str,
                          neg_train: str,
                          neg_test: str,
                          test_fraction: float = 0.2):
    """
    Read in equivalent phenotypes, split them into train/test (using test_fraction),
    then write out:
    pos_train (upheno_graph + train equivalent phenotypes)
    pos_test (test equivalent phenotypes)
    neg_train (random edges connecting nodes not connected in upheno_graph)
    neg_test (random edges connecting nodes not connected in equiv_phenotypes)

    :param upheno_graph file containing all edges from upheno (except equivalent
    phenotypes)
    :param equiv_phenotypes file containing equivalent phenotype edges, with weights
    :param weight_multiplier factor to multiply weight of phenotype links
    :param pos_train: filename write out pos train edges
    :param pos_test: filename to write out pos test edges
    :param neg_train: filename to write neg train edges
    :param neg_test: filename to write neg test edges
    :param test_fraction=0.2 what fraction of equiv_phenotypes should be used for
    testing

    :return: pos_train, pos_test, neg_train and neg_test graphs in CSFGraph format
    """

    curie_map = make_iri_to_curie_map()

    # write out pos_train and pos_test
    # first split equiv phenotype edges into train/test and write out positives edges
    logging.info("Making positive train and positive test files")
    with open(equiv_phenotypes.name, 'r') as equiv_fh, \
            open(pos_train, 'w') as pos_train_fh, \
            open(pos_test, 'w') as pos_test_fh:
        for line in equiv_fh:
            r = random.random()
            items = line.rstrip().split("\t")
            items[0] = curieize(items[0], curie_map)
            items[1] = curieize(items[1], curie_map)

            # default edge weight for known equivalent phenotypes
            if len(items) < 3:
                items.append("1")
            items[2] = str(float(items[2]) * weight_multiplier)

            outline = "\t".join(items) + "\n"
            if r > test_fraction:
                pos_train_fh.write(outline)
            else:
                pos_test_fh.write(outline)
        equiv_fh.close()
        pos_train_fh.close()
        pos_test_fh.close()

        # append upheno graph to pos_train edges:
        with open(pos_train, 'a') as pos_train_append_fh, \
                open(upheno_graph.name, 'r') as upheno_graph_fh:
            for line in upheno_graph_fh:
                # turn <IRI:1234> into CURIE:1234
                (item1, item2) = line.strip().split(" ")
                item1 = curieize(item1, curie_map)
                item2 = curieize(item2, curie_map)

                pos_train_append_fh.write("\t".join([item1, item2, "1"]) + "\n")

    logging.info("Loading CSFGraphs from positive train and positive test edge files")
    pos_train_graph = CSFGraph(pos_train)
    pos_test_graph = CSFGraph(pos_test)

    # make negative edges
    logging.info("Making negative training edges file")
    make_negative_edge_file(neg_train,
                            pos_train_graph.edge_count(),
                            pos_train_graph, pos_test_graph)

    logging.info("Making negative test edges file")
    make_negative_edge_file(neg_test,
                            pos_test_graph.edge_count(),
                            pos_train_graph, pos_test_graph)

    logging.info("Loading CSFGraphs from negative train and test edge file")
    return True


def make_negative_edge_file(filename: str,
                            num_edges_to_make: int,
                            pos_train_graph: CSFGraph,
                            pos_test_graph: CSFGraph) -> bool:
    edge_count = 0
    with open(filename, 'w') as neg_train_fh,\
            tqdm(total=pos_train_graph.edge_count()) as pbar:
        while edge_count < num_edges_to_make:
            node1_name = pos_train_graph.index_to_node_map[random_node(pos_train_graph)]
            node2_name = pos_train_graph.index_to_node_map[random_node(pos_train_graph)]

            if edge_count % 10 == 0:
                pbar.update(100)

            if not pos_train_graph.has_edge(node1_name, node2_name) and \
               not pos_test_graph.has_edge(node1_name, node2_name):
                neg_train_fh.write("\t".join([node1_name, node2_name, "1"]) + "\n")
                edge_count = edge_count + 1
    return True


def random_node(graph: CSFGraph) -> int:
    return int(numpy.random.uniform(0, graph.node_count(), 1))


def main(args):
    """
    The input files are positive training, positive test, negative training and negative test edges. The code
    reads the files and create graphs in CSFGraph format. Then, the positive training graph is embedded.
    Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Result of link prediction
    """
    print("[INFO]: p={}, q={}, classifier= {}, useGamma={}, word2vec_model={}".format(
        args.p, args.q, args.classifier, args.useGamma, args.w2v_model))

    pos_train = os.path.join("data/pos_train.edges")
    pos_test = os.path.join("data/pos_test.edges")
    neg_train = os.path.join("data/neg_train.edges")
    neg_test = os.path.join("data/neg_test.edges")

    if args.make_edge_files:
        logging.info("Remaking edge files")
        make_train_test_files(upheno_graph=args.upheno_graph,
                              equiv_phenotypes=args.equivalent_phenotypes,
                              weight_multiplier=args.weight_multiplier,
                              pos_train=pos_train,
                              pos_test=pos_test,
                              neg_train=neg_train,
                              neg_test=neg_test)
    else:
        logging.info("Using existing edge files")

    pos_train_graph = CSFGraph(pos_train)
    pos_test_graph = CSFGraph(pos_test)
    neg_train_graph = CSFGraph(neg_train)
    neg_test_graph = CSFGraph(neg_test)

    pos_train_g = xn2v.hetnode2vec.N2vGraph(pos_train_graph, args.p, args.q, args.gamma,
                                            args.useGamma)
    walks = pos_train_g.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, pos_train_graph, args.w2v_model)
    linkpred(pos_train_graph, pos_test_graph, neg_train_graph, neg_test_graph)


if __name__ == "__main__":
    args = parse_args()
    main(args)
