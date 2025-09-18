import copy
import json
import os  # 必须先于 torch 导入
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---- 关键修复 1 ----
bad_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if bad_alloc_conf and "=" in bad_alloc_conf:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

import torch
import torch.distributed as dist
import mmengine
import cv2   # <<< 新增

from pytorch_lightning.cli import LightningCLI

from no_time_to_train.models.hotfix_reset_cache import apply_patch as _apply_hotfix
_apply_hotfix()

from no_time_to_train.pl_wrapper.sam2ref_pl import RefSam2LightningModel
from no_time_to_train.pl_wrapper.sam2matcher_pl import Sam2MatcherLightningModel


def collect_results_cpu(result_part, size=None, tmpdir=None):
    if not dist.is_initialized():
        return result_part
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            mmengine.mkdir_or_exist("/tmp/.mydist_test")
            tmpdir = tempfile.mkdtemp(dir="/tmp/.mydist_test")
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda")
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    mmengine.dump(result_part, os.path.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f"part_{i}.pkl")
            part_list.append(mmengine.load(part_file))
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        if size is not None:
            ordered_results = ordered_results[:size]
        shutil.rmtree(tmpdir)
        return ordered_results


class SAM2RefLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--out_path", default=None, type=str)
        parser.add_argument("--out_support_res", default=None, required=False, type=str)
        parser.add_argument("--out_neg_pkl", default=None, required=False, type=str)
        parser.add_argument("--out_neg_json", default=None, required=False, type=str)
        parser.add_argument("--export_result", default=None, type=str)
        parser.add_argument("--seed", default=None, type=int)
        parser.add_argument("--n_shot", default=None, type=int)
        parser.add_argument("--coco_semantic_split", default=None, type=str)
        # 新增 debug 参数
        parser.add_argument(
            "--debug_check",
            default=False,
            action="store_true",
            help="打印PKL/JSON并保存可视化"
        )
        parser.add_argument(
            "--debug_outdir",
            default=None,
            type=str,
            help="保存可视化结果的目录(默认=./debug_vis)"
        )

    def before_test(self):
        memory_bank_cfg = self.model.model_cfg["memory_bank_cfg"]
        if self.model.test_mode == "fill_memory":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length"]
        elif self.model.test_mode == "fill_memory_neg":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length_negative"]
            self.model.dataset_cfgs["fill_memory"]["root"] = self.model.dataset_cfgs["support"]["root"]
            self.model.dataset_cfgs["fill_memory"]["json_file"] = self.config.test.out_neg_json
            self.model.dataset_cfgs["fill_memory"]["memory_pkl"] = self.config.test.out_neg_pkl

    def after_test(self):
        if self.model.test_mode in ["fill_memory", "postprocess_memory", "fill_memory_neg", "postprocess_memory_neg"]:
            if self.config.test.out_path is None:
                raise RuntimeError("Need out_path to save checkpoint with memory bank")
            save_path = self.config.test.out_path
            self.trainer.save_checkpoint(save_path)
            print(f"[INFO] Checkpoint saved to {save_path}")

        elif self.model.test_mode in ["test", "test_support"]:
            results = copy.deepcopy(self.trainer.model.output_queue)
            results_all = collect_results_cpu(results, size=len(self.trainer.model.eval_dataset))

            if len(self.trainer.model.scalars_queue) > 0:
                scalars = copy.deepcopy(self.trainer.model.scalars_queue)
                scalars_all = collect_results_cpu(scalars, size=len(self.trainer.model.eval_dataset))
            else:
                scalars_all = None

            if not dist.is_initialized() or dist.get_rank() == 0:
                if scalars_all is not None:
                    with open("./scalars_all.pkl", "wb") as f:
                        pickle.dump(scalars_all, f)

                results_unpacked = []
                for results_per_img in results_all:
                    if results_per_img:
                        results_unpacked.extend(results_per_img)
                if self.config.test.export_result is not None:
                    with open(self.config.test.export_result, "w") as f:
                        json.dump(results_unpacked, f)

                # 评估
                if self.model.test_mode == "test":
                    output_name = ""
                    if self.config.test.coco_semantic_split is not None:
                        output_name += f"semantic_split_{self.config.test.coco_semantic_split}_"
                    if self.config.test.n_shot is not None and self.config.test.seed is not None:
                        output_name += f"{self.config.test.n_shot}shot_{self.config.test.seed}seed"
                    if self.config.test.export_result is not None:
                        output_name += f"_{Path(self.config.test.export_result).stem}"
                    self.trainer.model.eval_dataset.evaluate(results_unpacked, output_name=output_name)
                elif self.model.test_mode == "test_support":
                    self.trainer.model.eval_dataset.evaluate(results_unpacked)
                    with open(self.config.test.out_support_res, "wb") as f:
                        pickle.dump(results_unpacked, f)

                # ========== Debug 可视化 & 检查 ==========
                if self.config.test.debug_check:
                    print("\n[DEBUG] ===== Checking PKL and JSON =====")
                    try:
                        with open(self.model.dataset_cfgs["fill_memory"]["memory_pkl"], "rb") as f:
                            pkl_data = pickle.load(f)
                        for cid, samples in list(pkl_data.items())[:3]:
                            print(f"Category {cid}: {len(samples)} samples")
                            for s in samples[:2]:
                                print(s)
                    except Exception as e:
                        print(f"[DEBUG] PKL load failed: {e}")

                    try:
                        json_file = self.model.dataset_cfgs["test"]["json_file"]
                        with open(json_file, "r") as f:
                            jdata = json.load(f)
                        print(f"JSON images={len(jdata['images'])}, anns={len(jdata['annotations'])}")
                        for ann in jdata["annotations"][:5]:
                            print(ann)

                        # 保存路径改为 debug_outdir
                        out_dir = Path(self.config.test.debug_outdir or "./debug_vis")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[DEBUG] 保存可视化到 {out_dir}")

                        id2img = {im["id"]: im for im in jdata["images"]}
                        for ann in jdata["annotations"][:5]:
                            img_info = id2img[ann["image_id"]]
                            img_path = Path(self.model.dataset_cfgs["test"]["root"]) / img_info["file_name"]
                            if not img_path.exists():
                                continue
                            img = cv2.imread(str(img_path))
                            segms = ann.get("segmentation", [])
                            if segms:
                                for seg in segms:
                                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                                    cv2.polylines(img, [pts], True, (0, 0, 255), 2)
                            else:
                                x, y, w, h = ann["bbox"]
                                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                            save_p = out_dir / f"ann_{ann['id']}.jpg"
                            cv2.imwrite(str(save_p), img)
                            print(f"[DEBUG VIS] saved {save_p}")
                    except Exception as e:
                        print(f"[DEBUG] JSON check failed: {e}")

        elif self.model.test_mode == "vis_memory":
            pass
        else:
            raise NotImplementedError(f"Unrecognized test mode {self.model.test_mode}")


if __name__ == "__main__":
    SAM2RefLightningCLI()
