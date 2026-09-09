"""The sequential-design framework: four ABCs, the values between them, and
the loop that drives them.

Implement :class:`Task` to plug your own screens in, :class:`Model` and
:class:`AcquisitionFunction` to plug your own method in, :class:`Metric` to
score it your own way, then hand them to :class:`SequentialLoop`. Nothing
here touches the network or the filesystem.
"""

from .acquisition import AcquisitionFunction
from .loop import SequentialLoop
from .metric import Metric
from .model import Model
from .task import Task
from .types import (
    HistoryEntry,
    ModelPrediction,
    Observation,
    RunResult,
    StepRecord,
)

__all__ = [
    "Task",
    "Model",
    "AcquisitionFunction",
    "Metric",
    "SequentialLoop",
    "Observation",
    "ModelPrediction",
    "StepRecord",
    "RunResult",
    "HistoryEntry",
]
