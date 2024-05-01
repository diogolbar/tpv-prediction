from kedro.pipeline import Pipeline, node, pipeline

from .nodes import optimize_model,train_model, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   node(
                func=optimize_model,
                inputs = ["X_train", "X_val","y_train", "y_val","params:model_options"],
                outputs = "hyperparameters",
                name = "optimize_model_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "hyperparameters"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
            func=predict,
            inputs=["classifier", "X_val"],
            outputs=["predictions","predictions_proba"],
            name="predict_node"
            ),
        ]
    )