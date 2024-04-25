from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, optimize_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func=split_data,
                inputs=["processed_data", "params:model_options"],
                outputs=[ "X_train", "X_val", "y_train", "y_val"],
                name="split_data_node",
            ),
            node(
                func=optimize_model,
                inputs = ["X_train", "X_val","y_train", "y_val","params:model_options"],
                outputs = "hyperparameters",
                name = "optimize_model_node",
            ),
        ]
    )