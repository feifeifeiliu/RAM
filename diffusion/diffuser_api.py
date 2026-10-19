from diffusers import  DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler, EDMDPMSolverMultistepScheduler
import os
import torch
import yaml
import math
import tqdm
import time
from torch.cuda.amp import autocast

from model.tmr_model.losses import CosineSimilarityPenaltyLoss, InfoNCE_with_filtering, KLLoss, SelfRegularizationLoss
from utils.loss_util import masked_l2
from utils.scale_weight import calculate_weight
import matplotlib.pyplot as plt

class DiffusePipeline(object):
    
    def __init__(self, diff_config=None, train_config=None):
        self.config = diff_config
        self.device = 'cuda'
        self.torch_dtype = torch.float32
        self.diffusion_steps = diff_config.diffusion_steps
        self.prediction_type = diff_config.prediction_type
        self.init_loss_fn(train_config)
        self.p_sample_loop = self.generate_batch
        self.dynamic_cfg = True
        self.reg_start = 0.0
        self.reg_stop = 1.0
        self.multi_reg = False
        self.num_reg = 5
        
        # if self.torch_dtype == torch.float16:
        #     model = model.half()       
        # self.model = model.to(device)
        
        self.train_scheduler = DDPMScheduler(num_train_timesteps=self.diffusion_steps,
            beta_schedule=diff_config.beta_schedule,
            variance_type="fixed_small",
            prediction_type= diff_config.prediction_type,
            clip_sample=False)
        print('Training Diffusion_config:\n',self.train_scheduler.config)
    
    def init_inference(self, diffuser_name, num_inference_steps, dynamic_cfg=False):
        self.diffuser_name = diffuser_name
        self.num_inference_steps = num_inference_steps
        self.dynamic_cfg = dynamic_cfg
        # Load parameters from YAML file
        with open('configs/diffuser_params.yaml', 'r') as yaml_file:
            diffuser_params = yaml.safe_load(yaml_file)

        # Select diffusion'parameters based on diffuser_name
        if diffuser_name is not None:
            if diffuser_name in diffuser_params:
                params = diffuser_params[diffuser_name]
                scheduler_class_name = params['scheduler_class']
                additional_params = params['additional_params']
                # import pdb;pdb.set_trace()
                # align training parameters
                additional_params['num_train_timesteps'] = self.config.diffusion_steps
                additional_params['beta_schedule'] = self.config.beta_schedule
                additional_params['prediction_type'] = self.config.prediction_type

                try:
                    scheduler_class = globals()[scheduler_class_name]
                except KeyError:
                    raise ValueError(f"Class '{scheduler_class_name}' not found.")

                self.scheduler = scheduler_class(**additional_params)
            else:
                raise ValueError(f"Unsupported diffuser_name: {diffuser_name}")

    def generate_batch(self, model, shape, model_kwargs):

        # random sampling noise x_T
        sample = torch.randn(shape,device=self.device, dtype=self.torch_dtype)

        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [ torch.tensor([t] * shape[0], device=self.device).long() for t in self.scheduler.timesteps] 
        # import pdb; pdb.set_trace()
        if self.multi_reg:
            ml_list = []

        pre_pred = None
        reg_scale = model_kwargs['y']['reg_scale']
        cfg_scale = model_kwargs['y']['cfg_scale']
        trend = 'decrease'
        fn_type = 'cosine'
        x_min = 0
        x_max = self.diffusion_steps
        t_values=[]
        w_values=[]

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if self.dynamic_cfg:
                    # w_0 = calculate_weight(trend, fn_type, t[0].item(), x_min, x_max)
                    # w_1 = calculate_weight('increase', fn_type, t[0].item(), x_min, x_max)
                    # t_values.append(t[0].item())
                    # w_values.append(w)
                    # print(w)
                    # REG window driven by env var REG_WINDOW="lo-hi" (inclusive),
                    # default "1-6" preserves existing Tab. 6 first-6 behavior.
                    _reg_window = os.environ.get('REG_WINDOW', '1-6')
                    _lo, _hi = (int(_s) for _s in _reg_window.split('-'))
                    if _lo <= i <= _hi:
                        # print('hello')
                        w_0 = 1
                    else:
                        w_0 = 0
                    model_kwargs['y']['reg_scale'] = reg_scale * w_0
                    # model_kwargs['y']['reg_scale'] = cfg_scale * w_1

                # 1. model predict
                with autocast(enabled=True):
                    predict = model(sample, t, model_kwargs['y'])

                    # model_kwargs['y']['pre_pred'] = predict

                    if self._is_reg_enabled(t[0].item()) and (model_kwargs['y']['reg_scale'][0].item()>0):
                        # import pdb;pdb.set_trace()
                        motion_x_dict = {'x': predict.squeeze(2).permute(0,2,1), 'm_mask': model_kwargs['y']['mask'].squeeze(1).squeeze(1)}
                        if self.multi_reg:
                            ml_list.append(model.encode(motion_x_dict, sample_mean=True))
                            model_kwargs['y']['recon_latent'] = ml_list[:self.num_reg]
                        else:
                            model_kwargs['y']['recon_latent'] = model.encode(motion_x_dict, sample_mean=True)
                    elif 'recon_latent' in model_kwargs['y'].keys():
                        del model_kwargs['y']['recon_latent']

                # 2. compute less noisy motion and set x_t -> x_t-1
                sample = self.scheduler.step(predict, t[0], sample).prev_sample

        return sample

    def generate(self, caption, m_lens, batch_size=32, our_f=None):
        N = len(caption)
        infer_mode = ''
        if  getattr(self.model, 'cond_mask_prob', 0) > 0:
            infer_mode = 'classifier-free-guidance'
        print(f'\nUsing {self.diffuser_name} diffusion scheduler to {infer_mode} generate {N} motions, sampling {self.num_inference_steps} steps.')
        self.model.eval()

        all_output = []
        t_sum = 0
        cur_idx=0
        # import pdb;pdb.set_trace()
        for bacth_idx in tqdm.tqdm(range(math.ceil(N/batch_size))):
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
                if our_f is not None:
                    batch_our_f = our_f[cur_idx:]
                else:
                    batch_our_f = None
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
                if our_f is not None:
                    batch_our_f = our_f[cur_idx: cur_idx + batch_size]
                else:
                    batch_our_f = None
            torch.cuda.synchronize() 
            start_time = time.time()
            output = self.generate_batch(batch_caption, batch_m_lens, batch_our_f)
            torch.cuda.synchronize() 
            now_time = time.time()
            
            # The average inference time is calculated after GPU warm-up in the first 50 steps.
            if (bacth_idx+1) * self.num_inference_steps >= 50:
                t_sum += now_time-start_time

            # Crop motion with gt/predicted motion length
            B = output.shape[0]
            for i in range(B):
                all_output.append(output[i,:batch_m_lens[i]])

            cur_idx += batch_size

        # calcalate average inference time
        t_eval = t_sum/(bacth_idx-1)
        print('The average generation time of a batch motion (bs=%d) is %f seconds'%(batch_size,t_eval))
        return all_output, t_eval

    def training_losses(self, model, x_start, t_enc, t_dec, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # 1. Sample noise that we'll add to the motion
        real_noise = torch.randn_like(x_start)

        # 2. Sample a random timestep for each motion
        t = torch.randint(0, self.diffusion_steps, (x_start.shape[0],), device=self.device)

        # 3. Add noise to the motion according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        x_t = self.train_scheduler.add_noise(x_start, real_noise, t)
        
        cond = model_kwargs['y']
        if self.prediction_type =='sample':
            target = x_start
        elif self.prediction_type == 'epsilon':
            target = real_noise
        elif self.prediction_type == 'v_prediction':
            target = self.train_scheduler.get_velocity(x_start, real_noise, t)
        
        terms = {}
        
        text_x_dict = {'x_enc': cond['t_feature'], 't_enc': None, 'x_dec': x_t, 't_dec': t, 't_mask': cond['t_mask'], 'm_mask': cond['m_mask'], 'clip_emb': cond['clip_emb'],
                       'step': cond['step']}
        # text -> motion
        t_motions, t_latents, _, t_dists = model.train_forward(text_x_dict, return_all=True)
        
        terms["gen"] = masked_l2(target, t_motions, cond['m_mask']).mean()

        terms["data_mse"] = (
            terms["gen"] * self.loss_weight['sub_weight']['gen'] # L_gen: text -> motion
        )


        motion_x_dict = {'x_enc': x_start, 't_enc': None, 'x_dec': x_t, 't_dec': t, 't_mask': cond['t_mask'], 'm_mask': cond['m_mask'],'step': cond['step']}
        # motion -> motion
        m_motions_list, m_latents, _, m_dists = model.train_forward(motion_x_dict, return_all=True, sample_mean=True)


        # terms["m-m_mse"] = self.masked_l2(target, m_motions_list[0], cond['m_mask']).mean()
        terms["rec"] = masked_l2(target, m_motions_list, cond['m_mask']).mean()
        
        # VAE losses
        if model.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            terms["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )
            
        # Motion-centric latent alignment L_latent (paper Eq.6).
        # Detached (stop-grad) fraction (1-beta), learnable fraction beta.
        # NOTE: paper Eq.6 uses squared L2; kept as SmoothL1 to match released model.
        terms["latent"] = self.latent_loss_fn(t_latents, m_latents.detach()*(1-self.beta) + m_latents*self.beta)

        terms["csp"] = self.csp_loss_fn(m_latents, cond['sent_emb'])
        terms["sr"] = self.sr_loss_fn(m_latents, cond['sent_emb'])

        # Ablation variant (weight 0 in the paper's final model): TMR-style contrastive loss
        terms["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, cond['sent_emb'])

        terms["data_mse"] += (
            # + terms["m-m_mse"] * self.loss_weight['sub_weight']['m-m_mse'] # motion -> motion
            + terms["rec"] * self.loss_weight['sub_weight']['rec'] # L_rec: motion -> motion
        )

        # Weighted average of the losses
        terms["loss"] = sum(
            self.loss_weight[x] * val for x, val in terms.items() if x in self.loss_weight
        )

        return terms
    
    def init_loss_fn(self, train_config):
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=train_config['temperature'], threshold_selfsim=train_config['threshold_selfsim']
        )
        self.kl_loss_fn = KLLoss()
        self.csp_loss_fn = CosineSimilarityPenaltyLoss(
            temperature=train_config['temperature'], threshold_selfsim=train_config['threshold_selfsim']
        )
        self.sr_loss_fn = SelfRegularizationLoss(
            temperature=train_config['temperature'], threshold_selfsim=train_config['threshold_selfsim']
        )
        self.threshold_selfsim_metrics = train_config['threshold_selfsim_metrics']
        self.loss_weight = train_config['loss_weight']
        self.beta = train_config['beta']  # paper Eq.6 gradient modulator (legacy: md_rate = 1 - beta)

    def cfg_scheduler(self, cfg_scale, t, max_t, 
                      type='decrease', milestone=0, gamma=0.5):
        if t > milestone:
            if type == 'decrease':
                return cfg_scale
            else:
                return cfg_scale * gamma
        else:
            if type == 'decrease':
                return cfg_scale * gamma
            else:
                return cfg_scale
            
    def _is_reg_enabled(self, t) -> bool:

        reg_start_step = int(self.reg_start * self.diffusion_steps)
        reg_stop_step = int(self.reg_stop * self.diffusion_steps)
        is_within_range = reg_start_step < t < reg_stop_step

        return is_within_range
