import itertools
import json
import os
import shutil
import time
from math import factorial

from tqdm import trange, tqdm

from clusterapp.core import evaluate
from clusterapp.utils import build_library, format_table, std_out_err_redirect_tqdm


def export(names, labels_pred, filename):
    result = {}
    for name, label in zip(names, map(str, labels_pred)):
        category = result.get(label, [])
        category.append(name)
        result[label] = category
    with open(filename, 'w') as file:
        json.dump(result, file)


def log(line, log_file):
    print(line)
    log_file.write('%s\n' % line)


def mark_best_features(reports):
    best = {}
    for report in reports:
        for line in report:
            if line[1] == 'ALGORITHM':
                continue

            algorithm = line[1]
            score = float(line[2])
            b = best.get(algorithm, {
                'score': -2 ** 31,
                'line': line
            })
            best[algorithm] = b

            if score > b['score']:
                b['score'] = score
                b['line'][0] = ' '
                b['line'] = line
                line[0] = '*'


def report_algorithm(algorithm, X, labels_pred, labels_true, run_id):
    start = time.time()
    score = evaluate(X, labels_pred, labels_true)
    if CLASSIFIED:
        measures = ['AMI', 'ARI', 'Homogeneity', 'Completeness']
    else:
        measures = ['Silhouette', 'Calinski-Harabaz']
    return list(map(str, [
        '',
        algorithm,
        *[score[measure] for measure in measures],
        round(time.time() - start, 2),
        run_id
    ]))


def run(args):
    global LIBRARY, CLASSIFIED, EXPORT

    LIBRARY = build_library(args.path, args.classified)
    CLASSIFIED = args.classified
    EXPORT = args.export

    with open(args.config) as config:
        config = json.load(config)

    features = config.get('features')
    n_clusters = config.get('n_clusters', 2)
    categories = config.get('categories', getattr(LIBRARY, 'categories', None))
    export_path = config.get('export_path', '')

    shutil.rmtree(export_path, ignore_errors=True)
    os.makedirs(export_path, exist_ok=True)

    test(
        features_set=features,
        min_features=config.get('min_features', 1),
        max_features=config.get('max_features', len(features)),
        algorithms=config.get('algorithms'),
        categories=(categories or n_clusters),
        export_path=export_path
    )


def test(features_set, min_features, max_features, algorithms, categories, export_path):
    n = len(features_set)
    reports = []
    features_combinations = []
    i = 0

    with std_out_err_redirect_tqdm() as orig_stdout:
        sizes = trange(
            min_features,
            max_features + 1,
            desc='Checking subsets of features',
            file=orig_stdout,
            dynamic_ncols=True
        )
        for r in sizes:
            k = factorial(n) / (factorial(r) * factorial(n - r))
            combinations = tqdm(
                itertools.combinations(features_set, r),
                total=int(k),
                desc='Checking subsets of size %d' % r,
                file=orig_stdout,
                dynamic_ncols=True,
                leave=False
            )
            for features in combinations:
                if CLASSIFIED:
                    report = [
                        (' ', 'ALGORITHM', 'AMI', 'ARI', 'HOMOGENEITY', 'COMPLETENESS', 'TIME', 'ID')
                    ]
                else:
                    report = [
                        (' ', 'ALGORITHM', 'SILHOUETTE', 'CALINSKI-HARABAZ', 'TIME', 'ID')
                    ]
                for algorithm in algorithms:
                    try:  # TODO remove this try-except
                        X, scaled_X, names, labels_pred, labels_true = LIBRARY.predict(
                            categories=categories,
                            features=features,
                            algorithm=algorithm
                        )
                    except:
                        print('Error extracting features: %s' % str(features))
                        continue
                    report.append(report_algorithm(algorithm, scaled_X, labels_pred, labels_true, i))
                    if EXPORT:
                        export(
                            names,
                            labels_pred,
                            os.path.join(export_path, '%d [%s] %s.json' % (i, algorithm, '+'.join(features)))
                        )
                    i += 1
                reports.append(report)
                features_combinations.append(features)

    print()
    mark_best_features(reports)
    with open(os.path.join(export_path, 'log.txt'), 'w') as log_file:
        for features, report in zip(features_combinations, reports):
            log('%s' % str(features), log_file)
            log(format_table(report), log_file)
