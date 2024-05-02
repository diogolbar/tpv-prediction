from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_classification_dashboard


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_classification_dashboard,
                inputs=["classifier", "X_val", "y_val"],
                outputs="evaluation_plot",
                name="node_model_evaluation",
            ),
        ]
    )
