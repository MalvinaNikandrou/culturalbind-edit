from ..common.registry import Registry
from .crope import CropeDataset

Registry.register_dataset("crope", CropeDataset)
