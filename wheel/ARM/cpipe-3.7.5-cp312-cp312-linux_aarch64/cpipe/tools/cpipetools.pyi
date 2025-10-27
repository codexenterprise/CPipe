from cpipe.module.security import AESHelper as AESHelper, Security as Security, cp_m_i as cp_m_i, cp_m_p as cp_m_p

class CPipeTools:
    MODEL_TYPE_CODEX: int
    MODEL_TYPE_CPIPE: int
    @classmethod
    def encrypt_models(cls, model_path, license_password=None, license_path=None, model_type: int = ...): ...
