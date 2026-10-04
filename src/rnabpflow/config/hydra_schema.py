from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class FilteringConfig:
    max_len: int = MISSING
    min_len: int = MISSING


@dataclass
class DataConfig:
    csv_path: str = MISSING
    filtering: FilteringConfig = MISSING
    min_t: float = MISSING
    samples_per_eval_length: int = MISSING
    num_eval_lengths: int = MISSING
    batch_size: int = MISSING
    num_batch_size: int = MISSING
    max_squared_res: int = MISSING
    max_num_res_squared: int = MISSING
    eval_batch_size: int = MISSING
    num_workers: int = MISSING
    prefatch_factor: int = MISSING
    contact_distance_threshold: float = MISSING
    contact_minimum_sequence_separation: int = MISSING


@dataclass
class RotationConfig:
    train_schedule: str = MISSING
    sample_schedule: str = MISSING
    exp_rate: int = MISSING


@dataclass
class TranslationConfig:
    train_schedule: str = MISSING
    sample_schedule: str = MISSING


@dataclass
class SamplingConfig:
    num_timesteps: int = MISSING


@dataclass
class InterpolantConfig:
    min_t: float = MISSING
    rots: RotationConfig = MISSING
    trans: TranslationConfig = MISSING
    sampling: SamplingConfig = MISSING

    self_condition: bool = MISSING


@dataclass
class NodeConfig:
    c_s: int = MISSING
    c_pos_emb: int = MISSING
    c_timestep_emb: int = MISSING
    c_token_emb: int = MISSING
    embed_diffuse_mask: bool = MISSING
    max_num_res: int = MISSING
    timestep_int: int = MISSING


@dataclass
class EdgeConfig:
    single_bias_transition_n: int = MISSING
    c_s: int = MISSING
    c_p: int = MISSING
    relpos_k: int = MISSING
    use_rbf: bool = MISSING
    num_rbf: int = MISSING
    feat_dim: int = MISSING
    num_bins: int = MISSING
    self_condition: bool = MISSING


@dataclass
class IPAConfig:
    c_s: int = MISSING
    c_z: int = MISSING
    c_hidden: int = MISSING
    no_heads: int = MISSING
    no_qk_points: int = MISSING
    no_v_points: int = MISSING
    seq_tfmr_num_heads: int = MISSING
    seq_tfmr_num_layers: int = MISSING
    num_blocks: int = MISSING
    ss: int = MISSING
    dropout: float = MISSING


@dataclass
class ModelConfig:
    node_embed_size: int = MISSING
    edge_embed_size: int = MISSING
    symmetric: bool = MISSING
    node_features: NodeConfig = MISSING
    edge_features: EdgeConfig = MISSING
    ipa: IPAConfig = MISSING


@dataclass
class BatchOptimalTransportConfig:
    enabled: bool = MISSING
    cost: str = MISSING
    noise_per_sample: int = MISSING
    permute: bool = MISSING


@dataclass
class TrainingConfig:
    min_plddt_mask: int = MISSING
    loss: str = MISSING
    bb_atom_scale: float = MISSING
    trans_scale: float = MISSING
    translation_loss_weight: float = MISSING
    t_normalize_clip: float = MISSING
    rotation_loss_weights: float = MISSING
    aux_loss_weight: float = MISSING
    aux_loss_t_pass: float = MISSING
    tors_loss_scale: float = MISSING
    token_loss_weight: float = MISSING


@dataclass
class WandbConfig:
    name: str = MISSING
    project: str = MISSING
    save_code: bool = MISSING
    tags: list = MISSING
    mode: str = MISSING


@dataclass
class OptimizerConfig:
    lr: float = MISSING


@dataclass
class TrainerConfig:
    overfit_batches: int = MISSING
    min_epochs: int = MISSING
    max_epochs: int = MISSING
    accelerator: str = MISSING
    log_every_n_steps: int = MISSING
    deterministic: bool = MISSING
    strategy: str = MISSING
    check_val_every_n_epoch: int = MISSING
    accumulate_grad_batches: int = MISSING


@dataclass
class CheckpointerConfig:
    dirpath: str = MISSING
    save_last: bool = MISSING
    save_top_k: int = MISSING
    monitor: str = MISSING
    mode: str = MISSING
    every_n_epochs: int = MISSING
    filename: str = MISSING
    auto_insert_metric_name: bool = MISSING


@dataclass
class ExperimentConfig:
    finetune: bool = MISSING
    num_devices: int = MISSING
    warm_start: str | None = MISSING
    warm_start_config_override: bool = MISSING
    use_swa: bool = MISSING

    batch_ot: BatchOptimalTransportConfig = MISSING
    training: TrainingConfig = MISSING
    wandb: WandbConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    trainer: TrainerConfig = MISSING
    checkpointer: CheckpointerConfig = MISSING


@dataclass
class HydraConfig:
    data_cfg: DataConfig = MISSING
    interpolant: InterpolantConfig = MISSING
    model: ModelConfig = MISSING
    experiment: ExperimentConfig = MISSING
