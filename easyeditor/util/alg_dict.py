from ..dataset import CropeDataset
from ..models.dpo import apply_dpo_to_multimodal_model
from ..models.ft import apply_ft_to_multimodal_model
from ..models.grace import apply_grace_to_multimodal_model
from ..models.ike import apply_ike_to_multimodal_model
from ..models.mend import MendMultimodalRewriteExecutor
from ..models.wise import apply_wise_to_multimodal_model

ALG_MULTIMODAL_DICT = {
    "MEND": MendMultimodalRewriteExecutor().apply_to_model,
    "IKE": apply_ike_to_multimodal_model,
    "FT": apply_ft_to_multimodal_model,
    "DPO": apply_dpo_to_multimodal_model,
    "WISE": apply_wise_to_multimodal_model,
    "GRACE": apply_grace_to_multimodal_model,
}

MULTIMODAL_DS_DICT = {
    "crope": CropeDataset,
}
