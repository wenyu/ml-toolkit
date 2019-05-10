#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K
KB = K.backend

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def freeze_keras_model(model):
    session = KB.get_session()
    output_names = [v.op.name for v in model.outputs]
    return freeze_session(session, output_names=output_names)

def save_frozen_model(path, graph):
    tf.train.write_graph(graph, os.path.dirname(path), os.path.basename(path), as_text=False)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Convert a Keras model to frozen Tensorflow model.")
    parser.add_argument("keras_model_path",
                        nargs=1, type=str, help="Keras model path.")
    parser.add_argument("tf_model_path",
                        nargs=1, type=str, help="Tensorflow frozen model output path.")
    args = parser.parse_args()
    k_path = os.path.abspath(os.path.expanduser(args.keras_model_path[0]))
    t_path = os.path.abspath(os.path.expanduser(args.tf_model_path[0]))
    print(k_path, "=>", t_path)
    KB.set_learning_phase(0)
    model = K.models.load_model(k_path, compile=False)
    model.summary()
    frozen_graph = freeze_keras_model(model)
    save_frozen_model(t_path, frozen_graph)
