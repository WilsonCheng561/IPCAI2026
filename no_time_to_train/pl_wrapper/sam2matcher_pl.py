# no_time_to_train/pl_wrapper/sam2matcher_pl.py
import copy
from typing import Optional

import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule

from no_time_to_train.models.Sam2Matcher import Sam2Matcher
from no_time_to_train.models.Sam2MatchingBaseline import Sam2MatchingBaseline
from no_time_to_train.models.Sam2MatchingBaseline_noAMG import Sam2MatchingBaselineNoAMG

from no_time_to_train.dataset.metainfo import METAINFO
from no_time_to_train.dataset.coco_ref_dataset import (
    COCORefTrainDataset,
    COCOMemoryFillDataset,
    COCORefTestDataset,
    COCORefOracleTestDataset,
    COCOMemoryFillCropDataset
)

class DummyDataset(Dataset):
    def __init__(self, length):
        super(DummyDataset, self).__init__()
        self.data = [0.0 for _ in range(length)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def get_dataset(dataset_cfg_in: dict, stage: str):
    if dataset_cfg_in is None:
        raise ValueError(f"[get_dataset] dataset_cfg for stage='{stage}' is None.")
    dataset_cfg = copy.deepcopy(dataset_cfg_in)
    dataset_name = dataset_cfg.get("name", None)
    if dataset_name is None:
        raise ValueError(f"[get_dataset] dataset_cfg for stage='{stage}' has no 'name'. cfg={dataset_cfg}")
    if dataset_name not in ["coco", "cholec80"]:
        raise ValueError(f"[get_dataset] Unsupported dataset name '{dataset_name}'. "
                         f"Expected one of ['coco','cholec80'].")
    dataset_cfg.pop("name", None)
    if stage == "train":
        raise NotImplementedError
    if stage == "fill_memory":
        return COCOMemoryFillCropDataset(**dataset_cfg)
    elif stage == "vis_memory":
        dataset_cfg["custom_data_mode"] = "vis_memory"
        return COCOMemoryFillCropDataset(**dataset_cfg)
    elif stage == "fill_memory_neg":
        dataset_cfg["custom_data_mode"] = "fill_memory_neg"
        return COCOMemoryFillCropDataset(**dataset_cfg)
    elif stage == "test":
        return COCORefOracleTestDataset(**dataset_cfg)
    elif stage == "test_support":
        dataset_cfg["custom_data_mode"] = "test_support"
        return COCORefOracleTestDataset(**dataset_cfg)
    else:
        raise NotImplementedError(f"Unrecognized stage {stage}")

class Sam2MatcherLightningModel(LightningModule):
    def __init__(self, model_cfg: dict, dataset_cfgs: dict, data_load_cfgs: dict, test_mode: str = "none"):
        super().__init__()
        self.dataset_cfgs   = dataset_cfgs
        self.data_load_cfgs = data_load_cfgs
        self.workers        = data_load_cfgs.get("workers")
        # CLI 覆盖
        if "fill_memory.root" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["root"] = dataset_cfgs.pop("fill_memory.root")
        if "fill_memory.json_file" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["json_file"] = dataset_cfgs.pop("fill_memory.json_file")
        if "fill_memory.memory_pkl" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["memory_pkl"] = dataset_cfgs.pop("fill_memory.memory_pkl")
        if "fill_memory.memory_length" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["memory_length"] = int(dataset_cfgs.pop("fill_memory.memory_length"))
        if "fill_memory.cat_names" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["cat_names"] = dataset_cfgs.pop("fill_memory.cat_names").split(",")
        if "fill_memory.class_split" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["class_split"] = dataset_cfgs.pop("fill_memory.class_split")
            dataset_cfgs["fill_memory"]["cat_names"] = METAINFO[dataset_cfgs["fill_memory"]["class_split"]]
        if "memory_bank_cfg.length" in model_cfg:
            model_cfg["memory_bank_cfg"]["length"] = int(model_cfg.pop("memory_bank_cfg.length"))
        if "memory_bank_cfg.category_num" in model_cfg:
            model_cfg["memory_bank_cfg"]["category_num"] = int(model_cfg.pop("memory_bank_cfg.category_num"))
        if "dataset_name" in model_cfg:
            model_cfg["dataset_name"] = model_cfg.pop("dataset_name")
        if "test.imgs_path" in model_cfg:
            model_cfg["dataset_imgs_path"] = model_cfg.pop("test.imgs_path")
        if "test.online_vis" in model_cfg:
            model_cfg["online_vis"] = model_cfg.pop("test.online_vis")
        if "test.vis_thr" in model_cfg:
            model_cfg["vis_thr"] = float(model_cfg.pop("test.vis_thr"))
        if "test.root" in dataset_cfgs:
            dataset_cfgs["test"]["root"] = dataset_cfgs.pop("test.root")
        if "test.json_file" in dataset_cfgs:
            dataset_cfgs["test"]["json_file"] = dataset_cfgs.pop("test.json_file")
        if "test.cat_names" in dataset_cfgs:
            dataset_cfgs["test"]["cat_names"] = dataset_cfgs.pop("test.cat_names").split(",")
            model_cfg["class_names"] = dataset_cfgs["test"]["cat_names"]
        if "test.class_split" in dataset_cfgs:
            dataset_cfgs["test"]["class_split"] = dataset_cfgs.pop("test.class_split")
            dataset_cfgs["test"]["cat_names"] = METAINFO[dataset_cfgs["test"]["class_split"]]
        self.test_mode = test_mode
        self.model_cfg = copy.deepcopy(model_cfg)

        model_name = model_cfg.pop("name").lower()
        if model_name == "matcher":
            self.seg_model = Sam2Matcher(**model_cfg)
        elif model_name == "matching_baseline":
            self.seg_model = Sam2MatchingBaseline(**model_cfg)
        elif model_name == "matching_baseline_noamg":
            self.seg_model = Sam2MatchingBaselineNoAMG(**model_cfg)
        else:
            raise NotImplementedError(f"Unrecognized model name: {model_name}")

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def load_state_dict(self, state_dict, strict=False, assign=False):
        if state_dict is not None:
            super(Sam2MatcherLightningModel, self).load_state_dict(state_dict, strict=False, assign=assign)

    def _output_inqueue(self, output_one):
        if output_one is None:
            self.output_queue.append([])
            return
        if isinstance(output_one, (list, tuple)) and len(output_one) == 0:
            self.output_queue.append([])
            return
        if isinstance(output_one, (list, tuple)) and len(output_one) > 0 \
           and isinstance(output_one[0], dict) \
           and "image_id" in output_one[0] \
           and ("bbox" in output_one[0] or "segmentation" in output_one[0]):
            self.output_queue.append(list(output_one))
            return
        # 原始输出字典 → encode
        try:
            output_dict = output_one
            results_per_img = dict(
                masks=output_dict["binary_masks"].detach().cpu().numpy(),
                boxes=output_dict["bboxes"].detach().cpu().numpy(),
                scores=output_dict["scores"].detach().cpu().numpy(),
                labels=output_dict["labels"].detach().cpu().numpy(),
                img_id=int(output_dict["image_info"]["id"]) if str(output_dict["image_info"]["id"]).isdigit() else output_dict["image_info"]["id"]
            )
            encoded_list = self.eval_dataset.encode_results([results_per_img])
            self.output_queue.append(encoded_list)
            score_to_analysis = output_dict.get("score_to_analysis", None)
            if score_to_analysis is not None:
                self.scalars_queue.append(score_to_analysis.detach().cpu().numpy())
        except Exception:
            self.output_queue.append([])

    def forward(self, x, return_iou_grid_scores=False):
        return self.seg_model(x, return_iou_grid_scores)

    def test_step(self, batch, batch_idx):
        """
        - 不再 assert len(output) == len(batch)
        - 将 seg_model 输出统一整理为按样本 list（不足补空、超出截断）
        - 逐样本调用 _output_inqueue，保证 after_test 能顺利评估
        """
        assert not self.seg_model.training
        with torch.inference_mode():
            # 推断真实 batch 大小（默认 1）
            bs = 1
            try:
                if isinstance(batch, (list, tuple)):
                    bs = len(batch)
                elif isinstance(batch, dict):
                    bs = 1
            except Exception:
                bs = 1

            if self.test_mode in ["fill_memory", "vis_memory", "fill_memory_neg"]:
                self.seg_model._reset()  # 添加重置调用
                self.seg_model(batch)
                return None

            elif self.test_mode in ["test_support", "test"]:
                self.seg_model._reset()  # 添加重置调用
                output = self.seg_model(batch)

                # 统一为长度 = bs 的列表
                if output is None:
                    out_list = [[] for _ in range(bs)]
                elif isinstance(output, (list, tuple)):
                    out_list = list(output)
                    if len(out_list) < bs:
                        out_list += [[] for _ in range(bs - len(out_list))]
                    elif len(out_list) > bs:
                        out_list = out_list[:bs]
                else:
                    out_list = [output] + [[] for _ in range(max(0, bs - 1))]

                for i in range(bs):
                    self._output_inqueue(out_list[i])
                return None

            elif self.test_mode == "postprocess_memory":
                self.seg_model._reset()  # 添加重置调用
                self.seg_model.postprocess_memory()
                return None

            elif self.test_mode == "postprocess_memory_neg":
                self.seg_model._reset()  # 添加重置调用
                self.seg_model.postprocess_memory_negative()
                return None

            else:
                raise NotImplementedError("Unrecognized test mode: %s" % self.test_mode)

    def setup(self, stage: str):
        if stage != "test" and stage != "predict":
            raise NotImplementedError
        self.output_queue = []
        self.scalars_queue = []
        if self.test_mode == "fill_memory":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "fill_memory")
        elif self.test_mode == "vis_memory":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "vis_memory")
        elif self.test_mode == "test_support":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("support"), "test_support")
        elif self.test_mode == "fill_memory_neg":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "fill_memory_neg")
        elif self.test_mode == "test":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("test"), "test")
        elif self.test_mode == "postprocess_memory":
            self.eval_dataset = DummyDataset(1)
        elif self.test_mode == "postprocess_memory_neg":
            self.eval_dataset = DummyDataset(1)
        else:
            raise NotImplementedError("Unrecognized test mode: %s" % self.test_mode)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=1,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda batch: batch
        )
