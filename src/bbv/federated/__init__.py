"""Federated learning core modules."""

from bbv.federated.fedavg import (
    FederatedClient,
    FederatedServer,
    FedAvgResult,
    build_client,
    build_server,
    train_federated,
)

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FedAvgResult",
    "build_client",
    "build_server",
    "train_federated",
]
