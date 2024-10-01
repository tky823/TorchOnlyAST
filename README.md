# TorchOnlyAST

Audio spectrogram transformer that depends only on PyTorch.

## Installation

```sh
pip install git+https://github.com/tky823/TorchOnlyAST.git
```

## Pretrained models

### Audio spectrogram transformer (AST)

```python
>>> import torch
>>> from torch_only_ast.models.ast import AST, MLPHead
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 512
>>> model = AST.build_from_pretrained("ast-base-stride10")
>>> print(model)
AST(
  (embedding): PositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
    (dropout): Dropout(p=0, inplace=False)
  )
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
  (aggregator): HeadTokensAggregator()
  (head): MLPHead(
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=768, out_features=527, bias=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 527])
>>> # remove pretrained head
>>> model.head = None
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 768])
>>> # set customized head to model
>>> embedding_dim = model.embedding.embedding_dim
>>> num_classes = 50
>>> head = MLPHead(embedding_dim, num_classes)
>>> model.head = head
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # or set customized head to build_from_pretrained
>>> model = AST.build_from_pretrained("ast-base-stride10", head=head)
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # remove aggregator and pretrained head
>>> model.aggregator = None
>>> model.head = None
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 602, 768])  # 1 [CLS], 1 [DIST], and 600 patches
```

### Self-supervised audio spectrogram transformer (SSAST) for (continual) pretraining

- Patch-based SSAST

```python
>>> import torch
>>> from torch_only_ast.models.ssast import MultiTaskSSASTMPM
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 1024
>>> model = MultiTaskSSASTMPM.build_from_pretrained("multitask-ssast-patch-base-400")
>>> print(model)
MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(
  (embedding): PositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
    (dropout): Dropout(p=0, inplace=False)
  )
  (masker): Masker()
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
  (reconstructor): MLP(
    (linear1): Linear(in_features=768, out_features=768, bias=True)
    (nonlinear): ReLU()
    (linear2): Linear(in_features=768, out_features=256, bias=True)
  )
  (classifier): MLP(
    (linear1): Linear(in_features=768, out_features=768, bias=True)
    (nonlinear): ReLU()
    (linear2): Linear(in_features=768, out_features=256, bias=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> reconstruction, classification = model(input)
>>> reconstruction_output, reconstruction_target, reconstruction_length = reconstruction
>>> classification_output, classification_target, classification_length = classification
>>> print(reconstruction_output.size(), reconstruction_target.size(), reconstruction_length.size())
torch.Size([4, 400, 256]) torch.Size([4, 400, 256]) torch.Size([4])  # 400 tokens are masked.
```

- Frame-based SSAST

```python
>>> import torch
>>> from torch_only_ast.models.ssast import MultiTaskSSASTMPM
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 1024
>>> model = MultiTaskSSASTMPM.build_from_pretrained("multitask-ssast-frame-base-400")
>>> print(model)
MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(
  (embedding): PositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(128, 2), stride=(128, 2))
    (dropout): Dropout(p=0, inplace=False)
  )
  (masker): Masker()
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
  (reconstructor): MLP(
    (linear1): Linear(in_features=768, out_features=768, bias=True)
    (nonlinear): ReLU()
    (linear2): Linear(in_features=768, out_features=256, bias=True)
  )
  (classifier): MLP(
    (linear1): Linear(in_features=768, out_features=768, bias=True)
    (nonlinear): ReLU()
    (linear2): Linear(in_features=768, out_features=256, bias=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> reconstruction, classification = model(input)
>>> reconstruction_output, reconstruction_target, reconstruction_length = reconstruction
>>> classification_output, classification_target, classification_length = classification
>>> print(reconstruction_output.size(), reconstruction_target.size(), reconstruction_length.size())
torch.Size([4, 400, 256]) torch.Size([4, 400, 256]) torch.Size([4])
```

### SSAST for finetuning

- Patch-based SSAST

```python
>>> import torch
>>> from torch_only_ast.models.ssast import SSAST
>>> from torch_only_ast.models.ast import HeadTokensAggregator, MLPHead
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 512
>>> model = SSAST.build_from_pretrained("multitask-ssast-patch-base-400")
>>> print(model)
SSAST(
  (embedding): PositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
    (dropout): Dropout(p=0, inplace=False)
  )
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 258, 768])  # 1 [CLS], 1 [DIST], and 256 patches
>>> # set customized aggregator and head to model
>>> embedding_dim = model.embedding.embedding_dim
>>> num_classes = 50
>>> aggregator = HeadTokensAggregator(insert_cls_token=True, insert_dist_token=True)
>>> head = MLPHead(embedding_dim, num_classes)
>>> model.aggregator = aggregator
>>> model.head = head
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # or set customized aggregator and head to build_from_pretrained
>>> model = SSAST.build_from_pretrained("multitask-ssast-patch-base-400", aggregator=aggregator, head=head)
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
```

- Frame-based SSAST

```python
>>> import torch
>>> from torch_only_ast.models.ssast import SSAST
>>> from torch_only_ast.models.ast import HeadTokensAggregator, MLPHead
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 512
>>> model = SSAST.build_from_pretrained("multitask-ssast-frame-base-400")
>>> print(model)
SSAST(
  (embedding): PositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(128, 2), stride=(128, 2))
    (dropout): Dropout(p=0, inplace=False)
  )
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 258, 768])
>>> # set customized aggregator and head to model
>>> embedding_dim = model.embedding.embedding_dim
>>> num_classes = 50
>>> aggregator = HeadTokensAggregator(insert_cls_token=True, insert_dist_token=True)
>>> head = MLPHead(embedding_dim, num_classes)
>>> model.aggregator = aggregator
>>> model.head = head
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # or set customized aggregator and head to build_from_pretrained
>>> model = SSAST.build_from_pretrained("multitask-ssast-frame-base-400", aggregator=aggregator, head=head)
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
```

### Patchout fast spectrogram transformer (PaSST)

```python
>>> import torch
>>> from torch_only_ast.models.passt import PaSST
>>> from torch_only_ast.models.ast import MLPHead
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 512
>>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa")
>>> print(model)
PaSST(
  (embedding): DisentangledPositionalPatchEmbedding(
    (conv2d): Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
    (dropout): Dropout(p=0, inplace=False)
  )
  (dropout): StructuredPatchout(frequency_drops=4, time_drops=40)
  (backbone): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (linear1): Linear(in_features=768, out_features=3072, bias=True)
        (dropout): Dropout(p=0, inplace=False)
        (linear2): Linear(in_features=3072, out_features=768, bias=True)
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (dropout1): Dropout(p=0, inplace=False)
        (dropout2): Dropout(p=0, inplace=False)
        (activation): GELU(approximate='none')
      )
    )
    (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  )
  (aggregator): HeadTokensAggregator()
  (head): MLPHead(
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=768, out_features=527, bias=True)
  )
)
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 527])
>>> # remove pretrained head
>>> model.head = None
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 768])
>>> # set customized head to model
>>> embedding_dim = model.embedding.embedding_dim
>>> num_classes = 50
>>> head = MLPHead(embedding_dim, num_classes)
>>> model.head = head
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # or set customized head to build_from_pretrained
>>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa", head=head)
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 50])
>>> # remove aggregator and pretrained head
>>> model.aggregator = None
>>> model.head = None
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 82, 768])  # Patchout is applied during training.
>>> model.eval()
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 602, 768])  # Patchout is not applied during evaluation.
```

## License
- Apache License, Version 2.0 **EXCEPT FOR WEIGHTS OF PRETRAINED MODELS**
- Weights for some of the pre-trained models are extracted from the official implementations. Their licenses follow the official implementations.
