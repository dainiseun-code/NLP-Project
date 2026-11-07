"""Public snapshot execution policy; no model imports or artifact access."""


def disabled_execution_message(command: str) -> str:
    return (
        f"{command} is disabled in this public snapshot: historical model execution "
        "requires withheld plans, authorization records and model artifacts. "
        "Public validation, software checks, figure generation and offline "
        "reproduce scoring of a supplied bundle remain available."
    )
