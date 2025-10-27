from _typeshed import Incomplete

class codeXVision:
    LOCK_IDX: Incomplete
    INSERT_CONTENT: Incomplete
    @classmethod
    def encrypt_model(cls, model_path, save_path=None) -> None: ...
    @classmethod
    def decrypt_model(cls, model_path, save_path=None, return_bytes: bool = False): ...
