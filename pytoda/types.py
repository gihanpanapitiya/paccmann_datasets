"""Type definitions."""
import torch
from typing import List, Callable, Any, Tuple, Union, Sequence  # noqa

Tokens = List[str]
Indexes = List[int]
SMILESTokenizer = Callable[[str], Tokens]
Files = Sequence[str]
FileList = List[str]
GeneList = List[str]
FeatureList = List[str]
TransformList = List[Callable[[Any], Any]]
DrugSensitivityData = Tuple[torch.tensor, torch.tensor, torch.tensor]
AnnotatedData = Tuple[torch.tensor, torch.tensor]
