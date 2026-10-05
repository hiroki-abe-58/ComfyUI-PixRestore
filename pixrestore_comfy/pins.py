"""Pinned upstream revisions and the SHA-256 of every file this integration uses.

Provenance of the expected digests:
- PixRestore-S weights: Hugging Face LFS sha256 reported by the HF API for VCLab-PolyU/PixRestore@a5fe719.
- config.json: our sha256 of the file downloaded at that revision; its git blob SHA-1 equals the HF blobId 7a7e7566.
- DINOv2 ViT-S/14 weights: our sha256 of the file at the official URL (Meta publishes no digest for it).
- Upstream code files: our sha256 of each file at the pinned commit; each file's git blob SHA-1 equals the one in
  `git ls-tree` of that commit (LF checkout).
"""

PIXRESTORE_REPO = "csslc/PixRestore"
PIXRESTORE_COMMIT = "909ca06e614cf72e218d52b4fa41077c603d872b"
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
HF_REPO = "VCLab-PolyU/PixRestore"
HF_REVISION = "a5fe719a0c517422cff9f7a4895ab303e2320d51"
DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"

# <models>/pixrestore layout (same layout as the Hugging Face repository, so the official CLI accepts it too)
MODEL_DIR = "pixrestore-s"
MODEL_FILES = {
    "pixrestore-s/config.json": ("978deb1b0aaa544c12355ffb8ffffee1fbf012493de876736aba2c32185d18f4", 2281),
    "pixrestore-s/clean_weights/ema_model.safetensors": ("13c10ca30fa865977de393457e731db9f2469159dbd69c85fb3c79b3edf06895", 128709340),
}
DINOV2_FILE = "dinov2_vits14_pretrain.pth"
DINOV2_WEIGHTS = ("b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9", 88283115)

PIXRESTORE_CODE_DIR = f"upstream/PixRestore-{PIXRESTORE_COMMIT[:7]}"
DINOV2_CODE_DIR = f"upstream/dinov2-{DINOV2_COMMIT[:7]}"

PIXRESTORE_CODE = {
    "pixrestore/__init__.py": "562150ba920bbd1b64b6f07baa0d6ffc3fd66cad295b0505bb871e04f7cbb257",
    "pixrestore/flow.py": "4212987fd6514494ec45fd7315d0a2be1b263ffadcde4ce2492f4266eb37f9ae",
    "pixrestore/vision.py": "81cca0775440f1b0f4069ea148589ad04ef57b52412e1585c0603f5dc8afaa40",
    "pixrestore/models/__init__.py": "57ee7934e076c58027fec3a889d026c00e8e41b96a60e830b3094b94199a4b41",
    "pixrestore/models/layers.py": "8936ccec1c77fbc13ee31e892ebd5b399e6150da2035d3dbfae9865429cefcf3",
    "pixrestore/models/patch_embed.py": "28b0b670ff05d8d47d5258847493ca4cc10cd34407e166f86b62ced605b42bdc",
    "pixrestore/models/pixel_dit.py": "4bf20b499144b9d10c7cc01b9499170714ea1c8f5305a1f4083c867eaf7da6d3",
    "pixrestore/models/rms_norm.py": "009c3df874c72f68db08e6ac1a2199213605d62918b7488a697102eb24ac65f5",
    "pixrestore/models/rope.py": "7b73661b9efa16a538079d55795641e0aed2dab5e05e71ab6be7675349a91c06",
    "pixrestore/models/swiglu.py": "2f5824e2dc9c2c8911a61ae3e62819590804a9c4dd00ea03f0979744de374abb",
}
DINOV2_CODE = {
    "dinov2/__init__.py": "0d2b87b7e71c7f7279ccc4b4f8b541d26071425284a0ea01a5a435fac4511cfe",
    "dinov2/hub/__init__.py": "2932e94a2d67be87b717c49468214c1127be7864aedf37f5891d26be7bc1437f",
    "dinov2/hub/backbones.py": "871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e",
    "dinov2/hub/utils.py": "579613e3d7b82c2a387eddc215ad9f076b8740f9618229628395fe5755dbd131",
    "dinov2/models/__init__.py": "b93329ce676f3eb0c9a7790d1b3b2b114a0855ea8dc76e32e262d1a7dc6c55ae",
    "dinov2/models/vision_transformer.py": "7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea",
    "dinov2/layers/__init__.py": "1b55deed39d5ab0b589bef421bbfe06f24a10cd590a0f8403234c9ee4e34109d",
    "dinov2/layers/attention.py": "79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d",
    "dinov2/layers/block.py": "60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636",
    "dinov2/layers/dino_head.py": "9fdb1fa61c0609dc0876f711f9d0d828c33bf9fd972aab3d9c968d44e16b98d1",
    "dinov2/layers/drop_path.py": "b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a",
    "dinov2/layers/layer_scale.py": "dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135",
    "dinov2/layers/mlp.py": "255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa",
    "dinov2/layers/patch_embed.py": "40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142",
    "dinov2/layers/swiglu_ffn.py": "e46d2948fb97e497cf991ff82ce30cb59b59fdf2e12a19568417113716b4f119",
}

# Values of pixrestore-s/config.json that this integration was verified with. Any other value is refused.
EXPECTED_CONFIG = {
    "resolution": 512, "patch_size": 8, "bottleneck_dim": 768, "hidden_size": 384, "depth": 12, "num_heads": 6,
    "mlp_ratio": 4, "encoder_dim_ratio": 3, "use_qknorm": True, "use_swiglu": True, "use_rope": True, "use_rmsnorm": True,
    "encoder_type": "dinov2s", "encoder_dim": 384, "encoder_input_size": 448, "encoder_layers": [1, 2, 4, 6, 8, 10],
    "feature_norm": "channel_rmsnorm", "gate_temperature": 1.0, "dino_layer_router_mode": "content",
    "use_dino_layer_router": True, "same_condition_layer_weights": False, "use_lq_degradation_token": True,
    "fixed_train_t": 1.0, "use_dino_gan": True, "mixed_precision": "bf16", "cfg_ratio": 0.0, "cfg_scale": 1.0,
    "use_aelq": True, "use_venc": True, "cond_strength_aelq_test": 1, "cond_strength_venc": 1, "aug_noise": False,
    "use_time_shift": False, "t_eps": 0.05,
}
