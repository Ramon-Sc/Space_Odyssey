"""
This file contains the code to create a the pipeline for the selected methods.
"""

def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.load(file)

def create_pipeline(segmentation_method, featurization_method):
    pipeline = {}
    pipeline['segmentation'] = segmentation_methods
    pipeline['featurization'] = featurization_methods
    return pipeline

def create_result_df(pipeline):
    pass

