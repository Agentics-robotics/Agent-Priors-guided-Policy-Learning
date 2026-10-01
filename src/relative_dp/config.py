"""Predeclared experiment settings. Freeze before formal optimization."""
TRAIN_CONFIG = dict(
    n_obs_steps=2, prediction_horizon=16, n_action_steps=4, action_dim=4,
    unet_down_dims=[64,128,256], diffusion_step_embed_dim=128,
    kernel_size=5, n_groups=8, batch_size=128, microbatch_size=128,
    optimizer="AdamW", learning_rate=0.0001, weight_decay=0.000001,
    betas=[0.9,0.999], max_grad_norm=1.0, train_updates=20000,
    lr_schedule="constant", diffusion_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2", prediction_type="epsilon",
    inference_scheduler="DDIM", inference_steps=16, ddim_eta=0.0,
    clip_predicted_clean_actions=True, ema_decay=0.995, train_seed=0,
    checkpoint_updates=[5000,10000,20000], obs_std_floor=0.001,
    cond_predict_scale=True, loader_seed=100, diffusion_seed=200,
    model_init_seed=0, precision="float32", torch_compile=True,
    train_compile_mode="reduce-overhead",
    inference_torch_compile=True, inference_compile_mode="reduce-overhead",
    beta_start=0.0001, beta_end=0.02, timestep_spacing="leading",
    steps_offset=0, set_alpha_to_one=True, rescale_betas_zero_snr=False,
    clip_sample_range=1.0, thresholding=False,
)
CONFIG = TRAIN_CONFIG
METAWORLD_COMMIT = "6e01ad7e2ffb2302e4dca04f796fcd8837df8540"
DP_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
