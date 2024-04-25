from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model


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
                func=train_model,
                inputs=["X_train", "y_train", "hyperparameters"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier", "X_val", "y_val"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )