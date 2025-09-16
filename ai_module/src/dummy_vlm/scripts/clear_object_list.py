#!/usr/bin/env python3
import argparse
import os
import sys


def default_object_list_path():
    # This script lives in <pkg>/scripts; object_list is in <pkg>/data/object_list.txt
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.abspath(os.path.join(this_dir, '..'))
    return os.path.join(pkg_dir, 'data', 'object_list.txt')


def clear_file(path: str):
    # Ensure directory exists; then truncate file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w'):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Clear (truncate) the object_list.txt file used by detection logs.')
    parser.add_argument('--path', '-p', default=default_object_list_path(), help='Path to object_list.txt (default: package data path)')
    args = parser.parse_args(argv)

    path = os.path.abspath(args.path)
    clear_file(path)
    print(f'Cleared: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
