from pathlib import Path
from urllib.request import urlretrieve

# Official ChangeFormer LEVIR-CD checkpoint release documented by the authors.
URL = "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip"

out = Path(__file__).parent / "changeformer_levir_release.zip"
print("Downloading official ChangeFormer LEVIR-CD release...")
urlretrieve(URL, out)
print("Saved:", out)
print("Unzip it and place the model checkpoint at models/changeformer_levir.pt, or adapt models/changeformer_adapter.py to the official repository structure.")
