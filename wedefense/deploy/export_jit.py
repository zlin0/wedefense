from __future__ import print_function

import argparse
import logging
import os

import torch
import yaml

from wedefense.models.get_model import get_model
from wedefense.models.projections import get_projection
from wedefense.deploy.MHFA import replace_gradmultiply_for_jit
from wedefense.frontend import frontend_class_dict
from wedefense.deploy.s3prl_jit.s3prl_frontend_jit_standalone import (
    JITCompatibleS3prlFrontendStandalone, )


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


class FrontendWrapper(torch.nn.Module):
    """Wrapper for frontend to handle JIT script export."""

    def __init__(self, frontend):
        super(FrontendWrapper, self).__init__()
        self.frontend = frontend

    def forward(self, input_wav, input_lengths):
        """Forward pass through frontend.

        Args:
            input_wav: Audio waveform tensor of shape [B, T]
            input_lengths: Length tensor of shape [B]

        Returns:
            Tuple of (features, lengths) where:
            - features: Feature tensor of shape [B, D, T, L] or [B, T, D]
            - lengths: Length tensor of shape [B]
        """
        return self.frontend(input_wav, input_lengths)


def main():
    args = get_args()
    args.jit = True
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    frontend = None
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"

        # Create original frontend first to load weights
        orig_frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])

        # Use JIT-compatible frontend for s3prl
        if frontend_type == "s3prl":
            upstream_name = configs['dataset_args'][frontend_args].get(
                'upstream_args', {}).get('name', '')
            if upstream_name in ['wav2vec2_large_960', 'xlsr_53']:
                # Will create JIT frontend later after loading orig_frontend weights
                frontend = None
                orig_frontend_for_jit = orig_frontend
            else:
                # Fallback to original frontend for unsupported models
                frontend = orig_frontend
        else:
            frontend = orig_frontend

        # Use orig_frontend to get output_size if frontend is None
        feat_dim_source = frontend if frontend is not None else (
            orig_frontend_for_jit
            if 'orig_frontend_for_jit' in locals() else orig_frontend)
        configs['model_args']['feat_dim'] = feat_dim_source.output_size()
        model = get_model(configs['model'])(**configs['model_args'])
        # model.add_module("frontend", frontend)
    else:
        model = get_model(configs['model'])(**configs['model_args'])

    projection = get_projection(configs['projection_args'])

    # Load checkpoint for backend and projection
    # Note: checkpoint contains keys like 'frontend.xxx', 'projection.xxx', and backend keys without prefix
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')

    # Extract backend weights (keys without 'frontend.' or 'projection.' prefix)
    backend_state = {}
    projection_state = {}

    for key, value in checkpoint_data.items():
        if key.startswith('frontend.'):
            # Skip frontend weights
            continue
        elif key.startswith('projection.'):
            # Projection weights - remove prefix
            new_key = key[11:]  # len('projection.') = 11
            projection_state[new_key] = value
        elif key in ['weights_k', 'weights_v'
                     ] or key.startswith('cmp_linear_') or key.startswith(
                         'att_head') or key.startswith('pooling_fc'):
            # Backend weights (no prefix)
            backend_state[key] = value

    # Load backend weights
    if backend_state:
        missing, unexpected = model.load_state_dict(backend_state,
                                                    strict=False)
        print(
            f"Loaded {len(backend_state)} backend parameters from checkpoint")
        if missing:
            print(f"Missing backend keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected backend keys: {len(unexpected)}")
    else:
        print("Warning: No backend weights found in checkpoint!")

    # Load projection weights
    if projection_state:
        missing, unexpected = projection.load_state_dict(projection_state,
                                                         strict=False)
        print(
            f"Loaded {len(projection_state)} projection parameters from checkpoint"
        )
        if missing:
            print(f"Missing projection keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected projection keys: {len(unexpected)}")
    else:
        print("Warning: No projection weights found in checkpoint!")

    # Load frontend weights from checkpoint and create JIT frontend
    if 'orig_frontend_for_jit' in locals(
    ) and orig_frontend_for_jit is not None:
        # Load weights into original frontend first
        # Reuse checkpoint_data loaded above
        frontend_state = {}
        for key, value in checkpoint_data.items():
            if key.startswith('frontend.'):
                new_key = key[9:]  # Remove 'frontend.' prefix
                frontend_state[new_key] = value

        if frontend_state:
            missing, unexpected = orig_frontend_for_jit.load_state_dict(
                frontend_state, strict=False)
            print(
                f"Loaded {len(frontend_state)} original frontend parameters from checkpoint"
            )
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
        else:
            print("Warning: No frontend weights found in checkpoint!")

        # Now create JIT frontend and load fine-tuned weights from original frontend
        upstream_name = configs['dataset_args'][frontend_args].get(
            'upstream_args', {}).get('name', '')
        print("Creating JIT frontend...")
        frontend = JITCompatibleS3prlFrontendStandalone(
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        print("Loading fine-tuned weights into JIT frontend...")
        frontend.load_fine_tuned_weights(orig_frontend_for_jit)
        print("JIT frontend created with fine-tuned weights")
    elif frontend is not None:
        # Load weights for non-JIT frontends
        # Reuse checkpoint_data loaded above
        frontend_state = {}
        for key, value in checkpoint_data.items():
            if key.startswith('frontend.'):
                new_key = key[9:]  # Remove 'frontend.' prefix
                frontend_state[new_key] = value

        if frontend_state:
            missing, unexpected = frontend.load_state_dict(frontend_state,
                                                           strict=False)
            print(
                f"Loaded {len(frontend_state)} frontend parameters from checkpoint"
            )
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
        else:
            print("Warning: No frontend weights found in checkpoint!")

    print(frontend)

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

        # Export frontend separately if it exists
        if frontend is not None:
            frontend.eval()
            # Generate frontend output filename from model output filename
            output_dir = os.path.dirname(args.output_file) if os.path.dirname(
                args.output_file) else '.'
            output_basename = os.path.basename(args.output_file)
            # Remove extension and add frontend suffix
            frontend_output_file = os.path.join(
                output_dir,
                os.path.splitext(output_basename)[0] + '_frontend.pt')

            # Wrap frontend for JIT script export
            frontend_wrapper = FrontendWrapper(frontend)
            frontend_wrapper.eval()

            with torch.no_grad():
                try:
                    script_frontend = torch.jit.script(frontend_wrapper)
                    script_frontend.save(frontend_output_file)
                    print('Export frontend successfully with script, see {}'.
                          format(frontend_output_file))
                except Exception as e:
                    logging.warning(
                        'Failed to export frontend with script: {}'.format(
                            str(e)))
                    logging.warning(
                        'Frontend export skipped. You may need to handle frontend separately.'
                    )


if __name__ == '__main__':
    main()
