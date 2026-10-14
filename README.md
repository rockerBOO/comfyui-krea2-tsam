# comfyui-krea2-tsam

T-SAM (Text Self-Attention-guided alignment) for Krea 2, from Fakhabi, "Text-Image Alignment in Diffusion Models: The Role of Attention Sink" (Purdue MS thesis, 2025), chapter 5.

At a few early steps the latent is nudged so the similarity matrix of image-to-text attention maps matches the text encoder's self-attention matrix.

## Nodes

- **Krea2 T-SAM Encode** (`CLIP`, text -> `CONDITIONING`, `TSAM_TARGET`): encodes normally and builds the text target `T` from Qwen3-VL attention. Use the same CLIP and prompt for the sampler's positive input and this node.
- **Krea2 T-SAM** (`MODEL`, `TSAM_TARGET` -> `MODEL`): wraps the diffusion model. On steps `start_pct`..`end_pct` it runs `iters_per_step` gradient updates on the latent and only touches the positive conditioning.

Inputs of `Krea2 T-SAM`: `alpha`, `gamma`, `start_pct`, `end_pct`, `iters_per_step`, `layer_start`/`layer_end` (DiT blocks the maps are averaged over), `smooth` (Gaussian smoothing of the 2D maps), `normalize_grad`.

## Notes

- The paper's defaults (`alpha=15`, `gamma=4`) barely move a Krea 2 latent (gradient norm ~0.01 vs latent norm ~400 at 768px). Expect to use a much larger `alpha` or `normalize_grad`; `alpha=1000` already changes the composition heavily.
- `T` rows sum to 1 over the lower triangle but `S` rows sum to 1 over all columns, so early rows cannot be matched. Kept as in the paper.
- The gradient step uses PyTorch attention and block-level checkpointing, stops after `layer_end`, and clones the weights autograd saves. Peak extra VRAM at 768px was about 1 GB on top of the resident model.
- It calls `comfy.model_management.synchronize()` after each gradient step. Without it the image comes out black with `--async-offload`.

## Tests

`./run_tests.sh` (CPU, tiny Krea 2 DiT, no checkpoint).
