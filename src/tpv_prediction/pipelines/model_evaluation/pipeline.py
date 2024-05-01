from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_classification_dashboard_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_classification_dashboard_node,
            inputs=["y_val", "predictions_proba", "params:labels_list" ],
            outputs="evaluation_plot",
            name="node_model_evaluation",
            ),
    ])