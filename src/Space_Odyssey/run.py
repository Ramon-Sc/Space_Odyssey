"""
This file contains launch script for the Space Odyssey project.
"""

import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Space Odyssey')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    return parser.parse_args()  

def main():
    args = parse_arguments()
    config = yaml.load(open(args.config))
    pipeline = create_pipeline(config)
    result_df = create_result_df(pipeline)



if __name__ == '__main__':
    main()