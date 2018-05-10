import itertools
import json
import os
import time

from clusterapp.core import evaluate
from clusterapp.utils import build_library, print_table


def export(names, labels_pred, filename):
    result = {}
    for name, label in zip(names, map(str, labels_pred)):
        category = result.get(label, [])
        category.append(name)
        result[label] = category
    with open(filename, 'w') as file:
        json.dump(result, file)


def report_algorithm(algorithm, X, labels_pred, labels_true):
    start = time.time()
    score = evaluate(X, labels_pred, labels_true)
    if CLASSIFIED:
        measures = ['ARI', 'AMI', 'Homogeneity', 'Completeness']
    else:
        measures = ['Silhouette', 'Calinski-Harabaz']
    return list(map(str, [
        algorithm,
        *[score[measure] for measure in measures],
        round(time.time() - start, 2)
    ]))


def run(args):
    global LIBRARY, CLASSIFIED, EXPORT

    LIBRARY = build_library(args.path, args.classified)
    CLASSIFIED = args.classified
    EXPORT = args.export

    with open(args.config) as config:
        config = json.load(config)
    features = config.get('features')

    test(
        features_set=features,
        min_features=config.get('min_features', 1),
        max_features=config.get('max_features', len(features)),
        algorithms=config.get('algorithms'),
        n_clusters=config.get('n_clusters', 2),
        export_path=config.get('export_path')
    )


def test(features_set, min_features, max_features, algorithms, n_clusters, export_path):
    for r in range(min_features, max_features + 1):
        for features in itertools.combinations(features_set, r):
            print()
            print(features)
            if CLASSIFIED:
                report = [
                    ('ALGORITHM', 'ARI', 'AMI', 'HOMOGENEITY', 'COMPLETENESS', 'TIME')
                ]
            else:
                report = [
                    ('ALGORITHM', 'SILHOUETTE', 'CALINSKI-HARABAZ', 'TIME')
                ]
            for algorithm in algorithms:
                X, scaled_X, names, labels_pred, labels_true = LIBRARY.predict(
                    categories=getattr(LIBRARY, 'categories', n_clusters),
                    features=features,
                    algorithm=algorithm
                )
                report.append(report_algorithm(algorithm, scaled_X, labels_pred, labels_true))
                if EXPORT:
                    os.makedirs(export_path, exist_ok=True)
                    export(names, labels_pred,
                           os.path.join(export_path, '[%s]%s.json' % (algorithm, '+'.join(features))))
            print_table(report)
