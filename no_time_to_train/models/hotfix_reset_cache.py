# no_time_to_train/models/hotfix_reset_cache.py
# 作用：清理 Sam2MatchingBaseline_noAMG 在推理时遗留的缓存，避免
# AssertionError: assert self.backbone_features is None

from typing import Any

def apply_patch():
    from no_time_to_train.models.Sam2MatchingBaseline_noAMG import Sam2MatchingBaselineNoAMG as Cls

    # 备份原始方法
    _orig_forward_sam = Cls._forward_sam
    _orig_forward_test = Cls.forward_test

    # 一处统一的缓存清理器（尽量兼容不同分支变量名）
    def _clear_cache(self: Any):
        cand_attrs = [
            "backbone_features", "_backbone_features",
            "image_embeddings", "_image_embeddings",
            "high_res_features", "_high_res_features",
            "predictor_image_embeddings", "predictor_image_pe",
            "predictor_high_res_features", "sam_image_embeddings",
            "sam_high_res_feats", "sam_low_res_feats",
            "cached_keys", "cached_values",
        ]
        for a in cand_attrs:
            if hasattr(self, a):
                try:
                    setattr(self, a, None)
                except Exception:
                    pass

    # 覆写 _forward_sam：前后都清一次缓存；遇到断言再重试一次
    def _patched_forward_sam(self, *args, **kwargs):
        _clear_cache(self)
        try:
            out = _orig_forward_sam(self, *args, **kwargs)
        except AssertionError:
            _clear_cache(self)
            out = _orig_forward_sam(self, *args, **kwargs)
        _clear_cache(self)
        return out

    # 覆写 forward_test：遇到断言/“No masks found”做兜底，返回空结果以便外层评测继续
    def _patched_forward_test(self, input_dicts, with_negative: bool = False):
        try:
            return _orig_forward_test(self, input_dicts, with_negative=with_negative)
        except AssertionError:
            _clear_cache(self)
            return _orig_forward_test(self, input_dicts, with_negative=with_negative)
        except ValueError as e:
            # 让评测不中断，由上层 _output_inqueue 处理 None
            if "No masks found" in str(e):
                return [None]
            raise

    # 真正打补丁
    Cls._forward_sam = _patched_forward_sam
    Cls.forward_test = _patched_forward_test
