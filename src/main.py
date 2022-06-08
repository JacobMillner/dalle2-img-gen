import torch
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer, OpenAIClipAdapter


def load_diffusion_model(dprior_path, device, clip_choice):

    loaded_obj = torch.load(str(dprior_path), map_location='cpu')

    if clip_choice == "ViT-B/32":
        dim = 512
    else:
        dim = 768

    prior_network = DiffusionPriorNetwork(
        dim=dim,
        depth=12,
        dim_head=64,
        heads=12,
        normformer=True
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter(clip_choice),
        image_embed_dim=dim,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
    ).to(device)

    diffusion_prior.load_state_dict(loaded_obj["model"], strict=True)

    diffusion_prior = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
    ).to(device)

    diffusion_prior.optimizer.load_state_dict(loaded_obj['optimizer'])
    diffusion_prior.scaler.load_state_dict(loaded_obj['scaler'])

    return diffusion_prior
