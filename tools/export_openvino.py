# Copyright 2023 cansik.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This file expect the following requirements:
# pip install openvino nncf

import argparse
import os
from pathlib import Path

import nncf
import openvino as ov
import torch
from openvino import Layout
from openvino._pyopenvino.preprocess import PrePostProcessor

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.util import cfg, load_config
from tools import export_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to openvino.",
    )
    parser.add_argument("--cfg_path", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    parser.add_argument(
        "--nncf", action="store_true", help="Run NNCF (Neural Network Compression Framework)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")

    # export onnx model
    save_dir = Path(cfg.save_dir)
    onnx_model_path = save_dir.joinpath(f"{save_dir.name}.onnx")
    export_onnx.main(cfg, model_path, str(onnx_model_path), input_shape)

    # convert onnx to openvino
    mean_values, scale_values = cfg.data.train.pipeline.normalize

    ov_model = ov.convert_model(onnx_model_path, input=[1, 3, input_shape[1], input_shape[0]])

    # add pre-processing step
    ppp = PrePostProcessor(ov_model)
    pp_input = ppp.input(0)
    pp_input.model().set_layout(Layout("NCHW"))
    pp_input.preprocess().mean(mean_values).scale(scale_values)
    ov_model = ppp.build()

    # save ov model
    ov_model_path = save_dir.joinpath(f"{save_dir.name}.xml")
    ov.save_model(ov_model, ov_model_path)

    print(f"exported openvino to {ov_model_path}")

    if args.nncf:
        print("running nncf")

        print("creating dataloader...")
        val_dataset = build_dataset(cfg.data.val, "val")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=False,
        )


        def transform_fn(data_item):
            images = [img.numpy() for img in data_item["img"]]
            images = [img.reshape((1, *img.shape)) for img in images]
            return images


        calibration_dataset = nncf.Dataset(val_dataloader, transform_fn)
        quantized_model = nncf.quantize(ov_model, calibration_dataset)

        ov_ptq_model_path = save_dir.joinpath(f"{save_dir.name}-ptq.xml")
        ov.save_model(quantized_model, ov_ptq_model_path)

        print(f"exported quantized openvino to {ov_ptq_model_path}")
