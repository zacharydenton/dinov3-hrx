# Third-party notices

## DINOv3 model

Built with DINOv3.

This project runs Meta's
[`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m),
pinned to revision `c93d816fc9e567563bc068f01475bec89cc634a6`, and
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
pinned to revision `5931719e67bbdb9737e363e781fb0c67687896bc`.

The pretrained model is governed by the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/),
separately from this repository's Apache-2.0 project code. Obtain access and
review the terms through the model page before downloading it. Packing weights
for GPU execution does not replace their model license.

Neither the original model files nor exported weights are distributed in this
repository or its Cargo package. Redistributing model material requires
following Meta's terms, including the agreement and attribution requirements.

## Hugging Face download source

The optional automatic download uses [`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m/tree/c93d816fc9e567563bc068f01475bec89cc634a6),
revision `c93d816fc9e567563bc068f01475bec89cc634a6`, file `model.safetensors`.
The validated file SHA-256 is `208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2`.
These are the original model bytes; hosting them on Hugging Face does not change
the model terms above. Weights are cached outside the Cargo package.

ViT-B automatic downloads use
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/tree/5931719e67bbdb9737e363e781fb0c67687896bc),
revision `5931719e67bbdb9737e363e781fb0c67687896bc`, file `model.safetensors`,
under the same model terms.

## Additional ViT checkpoints

The same DINOv3 terms apply to these additional LVD-1689M checkpoints:

- [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/114c1379950215c8b35dfcd4e90a5c251dde0d32),
  revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51),
  revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/tree/c807c9eeea853df70aec4069e6f56b28ddc82acc),
  revision `c807c9eeea853df70aec4069e6f56b28ddc82acc`.
- [facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m/tree/b80367753773648a6793235ab9c65cdbb029506f),
  revision `b80367753773648a6793235ab9c65cdbb029506f`.

ViT-7B uses `model.safetensors.index.json` and its six listed SafeTensors shards.
The other checkpoints use `model.safetensors`. All weights remain separately
downloaded artifacts outside this repository and its Cargo package.
