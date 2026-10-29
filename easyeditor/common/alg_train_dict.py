def get_alg_train_dict():
    from ..trainer.algs import MEND

    return {
        "MEND": MEND,
    }


# Lazy initialization to avoid circular imports
_ALG_TRAIN_DICT = None


def _get_alg_train_dict():
    global _ALG_TRAIN_DICT
    if _ALG_TRAIN_DICT is None:
        _ALG_TRAIN_DICT = get_alg_train_dict()
    return _ALG_TRAIN_DICT


# For backward compatibility
class ALG_TRAIN_DICT_Proxy:
    def __getitem__(self, key):
        return _get_alg_train_dict()[key]

    def __contains__(self, key):
        return key in _get_alg_train_dict()

    def get(self, key, default=None):
        return _get_alg_train_dict().get(key, default)

    def keys(self):
        return _get_alg_train_dict().keys()

    def values(self):
        return _get_alg_train_dict().values()

    def items(self):
        return _get_alg_train_dict().items()


ALG_TRAIN_DICT = ALG_TRAIN_DICT_Proxy()
