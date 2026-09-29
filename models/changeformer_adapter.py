"""
ChangeFormer adapter placeholder.

Why isolated?
The official ChangeFormer repository uses its own model/config/checkpoint
layout. Keeping this adapter separate prevents the Streamlit application from
silently claiming that a generic CNN or pixel-difference result is a
ChangeFormer prediction.

Official reference:
https://github.com/wgcban/ChangeFormer

Use the official LEVIR-CD V6 checkpoint and implementation, then implement:
    predict(t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray
returning a HxW float32 change probability map in [0,1].
"""
