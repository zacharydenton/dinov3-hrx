# Third-party notices

## DINOv3 model

Built with DINOv3.

This project runs Meta's
[`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m),
pinned to revision `c93d816fc9e567563bc068f01475bec89cc634a6`.

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
