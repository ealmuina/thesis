import argparse

import clusterapp.cli as cli
import clusterapp.web as web

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--classified', action='store_true')
    subparsers = parser.add_subparsers()

    parser_web = subparsers.add_parser('web')
    parser_web.add_argument('--host', default='127.0.0.1')
    parser_web.add_argument('--port', type=int, default=5000)
    parser_web.set_defaults(func=web.run)

    parser_cli = subparsers.add_parser('cli')
    parser_cli.add_argument('config')
    parser_cli.add_argument('--export', action='store_true')
    parser_cli.set_defaults(func=cli.run)

    args = parser.parse_args()
    args.func(args)
