"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import preprocess_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["raw_data", "params:preprocess_options"],
                outputs=["processed_data", "labelencoder"],
                name="preprocess_data_node",
            ),
            node(
                func=split_data,
                inputs=["processed_data", "params:preprocess_options"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data_node",
            ),
        ]
    )
