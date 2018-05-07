import argparse
import time

import clusterapp.web as web
from .core import ClassifiedLibrary, UnclassifiedLibrary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--classified', action='store_true')
    parser.add_argument('--cli', action='store_true')
    args = parser.parse_args()

    classified = args.classified
    start = time.time()
    if classified:
        library = ClassifiedLibrary(args.path)
    else:
        library = UnclassifiedLibrary(args.path)
    print('Features computed in %.3f seconds.' % (time.time() - start))

    if args.cli:
        pass
    else:
        web.run(args.host, args.port, library, classified)
