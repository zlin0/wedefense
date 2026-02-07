# Copyright (c) 2025 Chengdong Liang (liangchengdongd@qq.com)
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

from __future__ import print_function

import argparse
import logging
import os

import torch
import yaml

from wedefense.models.get_model import get_model
from wedefense.models.projections import get_projection
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.export.MHFA import replace_gradmultiply_for_jit


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', default=None, help='output file')
    args = parser.parse_args()
    return args


class BackendProjectionModel(torch.nn.Module):
    """Model with backend and projection, accepts features as input."""

    def __init__(self, backend, projection):
        super(BackendProjectionModel, self).__init__()
        self.backend = backend
        self.projection = projection

    def forward(self, features):
        outputs = self.backend(features)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        batch_size = embeds.shape[0]
        dummy_label = torch.zeros(batch_size,
                                  dtype=torch.long,
                                  device=embeds.device)
        x = self.projection(embeds, dummy_label)
        return x[0] if isinstance(x, tuple) else x


def main():
    args = get_args()
    args.jit = True
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = get_model(configs['model'])(**configs['model_args'])
    projection = get_projection(configs['projection_args'])

    load_checkpoint(model, args.checkpoint)

    model.eval()
    print(model)

    # Replace GradMultiply for JIT export compatibility
    model = replace_gradmultiply_for_jit(model)

    if args.output_file:
        # Export backend + projection model
        backend_proj_model = BackendProjectionModel(model, projection)
        backend_proj_model.eval()

        with torch.no_grad():
            script_model = torch.jit.script(backend_proj_model)
            script_model.save(args.output_file)
            print('Export model successfully with script, see {}'.format(
                args.output_file))


if __name__ == '__main__':
    main()
