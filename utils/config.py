import os
from omegaconf import OmegaConf, open_dict

SMPL_DATA_PATH = "./body_models/smpl"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10

def _normalize_legacy_config(config):
    """Translate legacy config keys to the paper-aligned names in place.

    Older checkpoints ship a `my_setting.yaml` written with the pre-paper
    vocabulary (contrastive_diffusion / mdm_ae / md_rate / sc / rot_mse / ...).
    This shim rewrites those keys so the rest of the codebase only ever sees
    the paper-aligned names. It is idempotent: a config already using the new
    keys passes through untouched.

    Note the value inversion for `md_rate`: the paper's Eq.6 gradient modulator
    is beta (=0.01), while the legacy code stored md_rate = 1 - beta (=0.99).
    """
    OmegaConf.set_struct(config, False)

    train_cfg = config.get('train_config', None)
    if train_cfg is not None:
        # md_rate (= 1 - beta) -> beta
        if 'md_rate' in train_cfg and 'beta' not in train_cfg:
            train_cfg['beta'] = 1.0 - float(train_cfg['md_rate'])
            del train_cfg['md_rate']

        lw = train_cfg.get('loss_weight', None)
        if lw is not None:
            for old, new in (('rot_mse', 'data_mse'), ('sc', 'sr')):
                if old in lw and new not in lw:
                    lw[new] = lw[old]
                    del lw[old]
            sub = lw.get('sub_weight', None)
            if sub is not None:
                for old, new in (('t2m_mse', 'gen'), ('m2m_mse', 'rec')):
                    if old in sub and new not in sub:
                        sub[new] = sub[old]
                        del sub[old]

    model_cfg = config.get('model_config', None)
    if model_cfg is not None:
        if 'contrastive_diffusion' in model_cfg and 'ram' not in model_cfg:
            model_cfg['ram'] = model_cfg['contrastive_diffusion']
            del model_cfg['contrastive_diffusion']
        ram_cfg = model_cfg.get('ram', None)
        if ram_cfg is not None and 'contrastive_learning' in ram_cfg \
                and 'use_motion_encoder' not in ram_cfg:
            ram_cfg['use_motion_encoder'] = ram_cfg['contrastive_learning']
            del ram_cfg['contrastive_learning']
        if 'mdm_ae' in model_cfg and 'decoder' not in model_cfg:
            model_cfg['decoder'] = model_cfg['mdm_ae']
            del model_cfg['mdm_ae']

    return config


def read_yaml_file(file_path):
    config = OmegaConf.load(file_path)

    # Map any legacy key names to the paper-aligned names (idempotent).
    _normalize_legacy_config(config)

    # 重新解析插值（重要！否则修改后的变量不会生效）
    OmegaConf.resolve(config)
    return config


def save_yaml_file(config, output_path):

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存配置到 YAML 文件
    OmegaConf.save(config=config, f=output_path)
    
    print(f"save yaml to: {output_path}")