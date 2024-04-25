"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func = preprocess_data,
                inputs = ["raw_data","params:model_options"],
                outputs = "processed_data",
                name = "preprocess_data_node",
            ),
            ]
    )
