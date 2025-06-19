import torch
import numpy as np

class DDIMSampler:
    def __init__(self, generator, num_train_timesteps=1000, eta=0.0):
        self.generator = generator
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta  # Stochasticity: 0 = deterministic (pure DDIM), > 0 adds noise like DDPM
        self.set_inference_timesteps(50)  # default

        self._register_noise_schedule()

    def _register_noise_schedule(self):
        betas = np.linspace(0.0001, 0.02, self.num_train_timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.alphas_cumprod[:-1]
        ])

    def set_inference_timesteps(self, steps):
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, steps, dtype=torch.long)

    def step(self, timestep, latents, model_output, stochasticity=None):
        # Here timestep is an integer timestep
        # latents: current sample at timestep timestep
        # model_output: predicted noise ε_θ(x_t, timestep)

        if stochasticity is not None:
            eta = stochasticity
        else:
            eta = self.eta

        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod_prev[timestep]
        beta_t = 1 - alpha_t

        # Predict x0
        pred_x0 = (latents - beta_t.sqrt() * model_output) / alpha_t.sqrt()

        # Equation 12 from DDIM paper:
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * beta_t).sqrt()
        noise = torch.randn_like(latents, generator=self.generator) if eta > 0 else 0.0

        # Compute the next sample
        latents_prev = (
            alpha_prev.sqrt() * pred_x0 +
            (1 - alpha_prev - sigma**2).sqrt() * model_output +
            sigma * noise
        )
        return latents_prev

    def add_noise(self, x_start, timestep):
        # DDIM is deterministic — normally you don’t need this, but for img2img:
        alpha = self.alphas_cumprod[timestep]
        noise = torch.randn_like(x_start, generator=self.generator)
        return alpha.sqrt() * x_start + (1 - alpha).sqrt() * noise
