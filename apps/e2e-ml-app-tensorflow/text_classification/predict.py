import os
import sys
sys.path.append(".")
from argparse import ArgumentParser
from argparse import Namespace
import collections
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import tokenizer_from_json

from text_classification import config
from text_classification import data
from text_classification import models
from text_classification import utils


def get_run_components(run_dir):
    # Load args
    config = utils.load_json(
        os.path.join(run_dir, 'config.json'))
    args = Namespace(**config)

    # Load tokenizers
    with open(os.path.join(run_dir, 'X_tokenizer.json'), 'r') as fp:
        X_tokenizer = tokenizer_from_json(json.load(fp))
    y_tokenizer = LabelEncoder()
    y_tokenizer.classes_ = np.load(os.path.join(
        run_dir, 'y_tokenizer.npy'), allow_pickle=True)

    # Load model
    model = models.TextCNN(
        embedding_dim=args.embedding_dim,
        vocab_size=len(X_tokenizer.word_index)+1,
        num_filters=args.num_filters, filter_sizes=args.filter_sizes,
        hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes_))
    model.summary(input_shape=(10,))  # build it
    model_path = os.path.join(run_dir, 'model/cp.ckpt')
    model.load_weights(model_path)

    # Conv output model
    conv_outputs_model = models.ConvOutputsModel(
        vocab_size=len(X_tokenizer.word_index)+1,
        embedding_dim=args.embedding_dim, filter_sizes=args.filter_sizes,
        num_filters=args.num_filters)
    conv_outputs_model.summary(input_shape=(10,))  # build it

    # Set weights
    conv_outputs_model.layers[0].set_weights(model.layers[0].get_weights())
    conv_layer_start_num = 1
    for layer_num in range(conv_layer_start_num, conv_layer_start_num + len(args.filter_sizes)):
        conv_outputs_model.layers[layer_num].set_weights(
            model.layers[layer_num].get_weights())

    return args, model, conv_outputs_model, X_tokenizer, y_tokenizer


def get_probability_distribution(y_prob, classes):
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results


def get_top_n_grams(tokens, conv_outputs, filter_sizes):
    # Process conv outputs for each unique filter size
    n_grams = {}
    for i, filter_size in enumerate(filter_sizes):

        # Identify most important n-gram for each filter's output
        popular_indices = collections.Counter(
            np.argmax(conv_outputs[i][0], axis=0))

        # Get corresponding text
        start = popular_indices.most_common(2)[-1][0]
        end = min(len(tokens), start+filter_size)
        n_gram = " ".join([token for token in tokens[start:end]])
        n_grams[filter_size] = {
            'n_gram': n_gram,
            'start': np.float64(start),
            'end': np.float64(end)
            }

    return n_grams


def predict(inputs, args, model, conv_outputs_model, X_tokenizer, y_tokenizer):
    """Predict the class for a text using
    a trained model from an experiment."""

    # Create dataset generator
    texts = [sample['text'] for sample in inputs]
    X = np.array(X_tokenizer.texts_to_sequences(texts))
    preprocessed_texts = X_tokenizer.sequences_to_texts(X)
    y_filler = np.array([0]*len(X))
    prediction_generator = data.DataGenerator(
        X=X, y=y_filler, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))

    # Predict
    results = []
    y_prob = model.predict(x=prediction_generator, verbose=1)
    conv_outputs = conv_outputs_model.predict(x=prediction_generator, verbose=1)
    for index in range(len(X)):
        results.append({
            'raw_input': texts[index],
            'preprocessed_input': preprocessed_texts[index],
            'probabilities': get_probability_distribution(y_prob[index], y_tokenizer.classes_),
            'top_n_grams': get_top_n_grams(tokens=preprocessed_texts[index].split(' '),
                                           conv_outputs=conv_outputs,
                                           filter_sizes=args.filter_sizes)})

    return results


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--text', type=str,
                        required=True, help="text to predict")
    args = parser.parse_args()
    inputs = [{'text': args.text}]

    # Get best run
    best_run = utils.get_best_run(project="GokuMohandas/e2e-ml-app-tensorflow",
                                  metric="test_loss", objective="minimize")

    # Load best run (if needed)
    best_run_dir = utils.load_run(run=best_run)

    # Get run components for prediction
    args, model, conv_outputs_model, X_tokenizer, y_tokenizer = get_run_components(
        run_dir=best_run_dir)

    # Predict
    results = predict(inputs=inputs, args=args, model=model,
                      conv_outputs_model=conv_outputs_model,
                      X_tokenizer=X_tokenizer, y_tokenizer=y_tokenizer)
    config.logger.info(json.dumps(results, indent=4, sort_keys=False))
