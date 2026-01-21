# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2025 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math


class MarginScheduler:

    def __init__(self,
                 model,
                 epoch_iter,
                 increase_start_epoch,
                 fix_start_epoch,
                 initial_margin,
                 final_margin,
                 update_margin,
                 increase_type='exp'):
        '''
        The margin is fixed as initial_margin before increase_start_epoch,
        between increase_start_epoch and fix_start_epoch, the margin is
        exponentially increasing from initial_margin to final_margin
        after fix_start_epoch, the margin is fixed as final_margin.
        '''
        self.model = model
        self.increase_start_iter = (increase_start_epoch - 1) * epoch_iter
        self.fix_start_iter = (fix_start_epoch - 1) * epoch_iter
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type

        self.fix_already = False
        self.current_iter = 0
        self.update_margin = update_margin and hasattr(self.model.projection,
                                                       'update')
        self.increase_iter = self.fix_start_iter - self.increase_start_iter

        self.init_margin()

    def init_margin(self):
        if hasattr(self.model.projection, 'update'):
            self.model.projection.update(margin=self.initial_margin)

    def get_increase_margin(self):
        initial_val = 1.0
        final_val = 1e-3

        current_iter = self.current_iter - self.increase_start_iter

        if self.increase_type == 'exp':  # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_iter / self.increase_iter) *
                math.log(final_val / (initial_val + 1e-6))) * initial_val
        else:  # linearly increase the margin
            ratio = 1.0 * current_iter / self.increase_iter
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def step(self, current_iter=None):
        if not self.update_margin or self.fix_already:
            return

        if current_iter is not None:
            self.current_iter = current_iter

        if self.current_iter >= self.fix_start_iter:
            self.fix_already = True
            if hasattr(self.model.projection, 'update'):
                self.model.projection.update(margin=self.final_margin)
        elif self.current_iter >= self.increase_start_iter:
            if hasattr(self.model.projection, 'update'):
                self.model.projection.update(margin=self.get_increase_margin())

        self.current_iter += 1

    def get_margin(self):
        try:
            margin = self.model.projection.margin
        except Exception:
            margin = 0.0

        return margin


class BaseClass:
    """Base class for learning rate scheduler with automatic frontend detection.
    
    This scheduler supports differential learning rates for frontend (e.g., pretrained
    SSL models like WavLM/HuBERT) and backend modules. Frontend modules typically use
    a smaller learning rate to preserve pretrained features.
    
    Features:
        - Automatic frontend detection via parameter names
        - Backward compatible with explicit 'name' field in param_groups
        - Supports multi-process warmup with scale_ratio
    """

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False,
                 frontend_modules=None,
                 frontend_lr_ratio=0.1,
                 model=None):
        """Initialize learning rate scheduler with frontend detection.
        
        Args:
            optimizer: PyTorch optimizer instance.
            num_epochs: Total number of training epochs.
            epoch_iter: Number of iterations per epoch.
            initial_lr: Initial learning rate.
            final_lr: Final learning rate at end of training.
            warm_up_epoch: Number of epochs for multi-process warmup (default: 6).
            scale_ratio: Learning rate multiplier for multi-process training (default: 1.0).
            warm_from_zero: If True, warmup from 0; if False, warmup from initial_lr (default: False).
            frontend_modules: List of module names to identify as frontend (default: ['frontend']).
                Examples: ['frontend'], ['wav2vec2', 'hubert'], ['module.frontend'] for DDP.
            frontend_lr_ratio: Learning rate ratio for frontend modules (default: 0.1).
                Frontend LR = base_lr * frontend_lr_ratio.
            model: Optional model reference for automatic parameter name detection.
                If provided, enables automatic frontend identification via parameter names.
                Note: Pass the original model, not DDP-wrapped model.
        """
        self.optimizer = optimizer
        self.max_iter = num_epochs * epoch_iter
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.scale_ratio = scale_ratio
        self.current_iter = 0
        self.warm_up_iter = warm_up_epoch * epoch_iter
        self.warm_from_zero = warm_from_zero
        self.frontend_lr_ratio = frontend_lr_ratio
        self.model = model
        
        # Set default frontend module names
        if frontend_modules is None:
            self.frontend_modules = ['frontend']
        elif isinstance(frontend_modules, str):
            self.frontend_modules = [frontend_modules]
        else:
            self.frontend_modules = list(frontend_modules)
        
        # Build parameter ID to name mapping (if model is provided)
        self._param_id_to_name = {}
        if self.model is not None:
            for name, param in self.model.named_parameters():
                self._param_id_to_name[id(param)] = name
        
        # Auto-restructure optimizer if single param group detected
        self._auto_restructure_optimizer()
        
        # Automatically identify and mark frontend parameter groups
        self._identify_frontend_param_groups()

    def _auto_restructure_optimizer(self):
        """Automatically restructure optimizer to separate frontend/backend params.
        
        This method detects if the optimizer has only a single parameter group
        and automatically splits it into frontend and backend groups if:
        1. Model reference is provided
        2. Frontend parameters can be identified
        3. Both frontend and backend parameters exist
        
        This allows using standard optimizer initialization (e.g., model.parameters())
        while still achieving differential learning rates.
        
        Note: This modifies optimizer.param_groups in-place.
        """
        # Only restructure if we have exactly one param group and a model
        if len(self.optimizer.param_groups) != 1 or self.model is None:
            return
        
        # Get the single param group
        original_group = self.optimizer.param_groups[0]
        all_params = original_group['params']
        
        # Separate frontend and backend parameters
        frontend_params = []
        backend_params = []
        
        for param in all_params:
            param_name = self._param_id_to_name.get(id(param), '')
            
            # Check if parameter belongs to frontend
            is_frontend_param = False
            for frontend_keyword in self.frontend_modules:
                if param_name.startswith(frontend_keyword + '.'):
                    is_frontend_param = True
                    break
            
            if is_frontend_param:
                frontend_params.append(param)
            else:
                backend_params.append(param)
        
        # Only restructure if we successfully separated params
        if len(frontend_params) > 0 and len(backend_params) > 0:
            # Create new param groups (preserve all original settings)
            frontend_group = {k: v for k, v in original_group.items() if k != 'params'}
            frontend_group['params'] = frontend_params
            
            backend_group = {k: v for k, v in original_group.items() if k != 'params'}
            backend_group['params'] = backend_params
            
            # Replace the single group with two groups
            self.optimizer.param_groups = [frontend_group, backend_group]
            
            # Log the restructuring (optional: can be enabled if needed)
            # print(f"✓ Auto-restructured optimizer: {len(frontend_params)} frontend, {len(backend_params)} backend params")

    def _identify_frontend_param_groups(self):
        """Automatically identify which parameter groups belong to frontend modules.
        
        Identification strategy (priority order):
        1. If param_group has 'name' field, check if it's in frontend_modules list
        2. If no 'name' field, check parameter names via model.named_parameters()
        3. Mark each param_group with '_is_frontend' flag (internal use only)
        
        This method adds an internal '_is_frontend' flag to each parameter group
        without modifying the original param_group structure.
        
        Time Complexity: O(n_groups * n_params_per_group) worst case, but typically fast
            since we only check the first parameter in each group.
        """
        for idx, param_group in enumerate(self.optimizer.param_groups):
            is_frontend = False
            
            # Strategy 1: Check explicit 'name' field (highest priority)
            if 'name' in param_group:
                if param_group['name'] in self.frontend_modules:
                    is_frontend = True
            else:
                # Strategy 2: Automatic identification via parameter names
                # Only need to check the first parameter (params in same group usually belong to same module)
                params = param_group.get('params', [])
                if params:
                    # Get the name of the first parameter (requires reverse lookup)
                    first_param = params[0] if isinstance(params, list) else next(iter(params))
                    param_name = self._get_param_name(first_param)
                    
                    if param_name:
                        # Check if parameter name contains frontend keyword
                        for frontend_keyword in self.frontend_modules:
                            if param_name.startswith(frontend_keyword + '.'):
                                is_frontend = True
                                break
            
            # Add internal marker (using private key to avoid modifying original structure)
            param_group['_is_frontend'] = is_frontend
    
    def _get_param_name(self, param):
        """Reverse lookup parameter name from parameter object.
        
        Uses the param_id -> name mapping built in __init__.
        
        Args:
            param: torch.nn.Parameter object.
            
        Returns:
            str: Parameter name (e.g., 'frontend.encoder.weight'), or '' if not found.
        """
        return self._param_id_to_name.get(id(param), '')

    def get_multi_process_coeff(self):
        lr_coeff = 1.0 * self.scale_ratio
        if self.current_iter < self.warm_up_iter:
            if self.warm_from_zero:
                lr_coeff = self.scale_ratio * self.current_iter / self.warm_up_iter
            elif self.scale_ratio > 1:
                lr_coeff = (self.scale_ratio -
                            1) * self.current_iter / self.warm_up_iter + 1.0

        return lr_coeff

    def get_current_lr(self):
        '''
        This function should be implemented in the child class
        '''
        return 0.0

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self):
        """Set learning rates with automatic differentiation for frontend modules.
        
        Learning rate assignment:
        - Frontend modules: lr = current_lr * frontend_lr_ratio (default: 0.1)
        - Other modules: lr = current_lr
        
        Compatibility: Supports two identification methods
        1. Explicit 'name' field (legacy, backward compatible)
        2. Automatic parameter name detection (new, via __init__)
        
        The frontend learning rate ratio is typically set to 0.1 for pretrained
        SSL models (WavLM, HuBERT) to preserve learned representations while
        allowing backend modules to adapt quickly to the new task.
        """
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            # Priority 1: Use internal _is_frontend marker (set by _identify_frontend_param_groups)
            if param_group.get('_is_frontend', False):
                param_group['lr'] = current_lr * self.frontend_lr_ratio
            else:
                # Priority 2: Fallback to explicit 'name' field check (backward compatibility)
                if 'name' in param_group and param_group['name'] in self.frontend_modules:
                    param_group['lr'] = current_lr * self.frontend_lr_ratio
                else:
                    param_group['lr'] = current_lr

    def step(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        self.set_lr()
        self.current_iter += 1

    def step_return_lr(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        current_lr = self.get_current_lr()
        self.current_iter += 1

        return current_lr


class ExponentialDecrease(BaseClass):

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False,
                 frontend_modules=None,
                 frontend_lr_ratio=0.1,
                 model=None):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio, warm_from_zero, 
                         frontend_modules, frontend_lr_ratio, model)

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        current_lr = lr_coeff * self.initial_lr * math.exp(
            (self.current_iter / self.max_iter) *
            math.log(self.final_lr / self.initial_lr))
        return current_lr


class TriAngular2(BaseClass):
    '''
    The implementation of https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 cycle_step=2,
                 reduce_lr_diff_ratio=0.5,
                 frontend_modules=None,
                 frontend_lr_ratio=0.1,
                 model=None):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio, False,
                         frontend_modules, frontend_lr_ratio, model)

        self.reduce_lr_diff_ratio = reduce_lr_diff_ratio
        self.cycle_iter = cycle_step * epoch_iter
        self.step_size = self.cycle_iter // 2

        self.max_lr = initial_lr
        self.min_lr = final_lr
        self.gap = self.max_lr - self.min_lr

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        point = self.current_iter % self.cycle_iter
        cycle_index = self.current_iter // self.cycle_iter

        self.max_lr = self.min_lr + self.gap * self.reduce_lr_diff_ratio**cycle_index

        if point <= self.step_size:
            current_lr = self.min_lr + (self.max_lr -
                                        self.min_lr) * point / self.step_size
        else:
            current_lr = self.max_lr - (self.max_lr - self.min_lr) * (
                point - self.step_size) / self.step_size

        current_lr = lr_coeff * current_lr

        return current_lr


def show_lr_curve(scheduler):
    import matplotlib.pyplot as plt

    lr_list = []
    for current_lr in range(0, scheduler.max_iter):
        lr_list.append(scheduler.step_return_lr(current_lr))
    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, '-o', markersize=1)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.show()


if __name__ == '__main__':
    optimizer = None
    num_epochs = 6
    epoch_iter = 500
    initial_lr = 0.6
    final_lr = 0.1
    warm_up_epoch = 2
    scale_ratio = 4
    scheduler = ExponentialDecrease(optimizer, num_epochs, epoch_iter,
                                    initial_lr, final_lr, warm_up_epoch,
                                    scale_ratio)
    # scheduler = TriAngular2(optimizer,
    #                         num_epochs,
    #                         epoch_iter,
    #                         initial_lr,
    #                         final_lr,
    #                         warm_up_epoch,
    #                         scale_ratio,
    #                         cycle_step=2,
    #                         reduce_lr_diff_ratio=0.5)

    show_lr_curve(scheduler)
