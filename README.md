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
>>> from torch_only_ast.models.ast import AudioSpectrogramTransformer
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 50
>>> model = AudioSpectrogramTransformer.build_from_pretrained("ast-base-stride10")
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 527])
```

### Patchout fast spectrogram transformer (PaSST)

```python
>>> import torch
>>> from torch_only_ast.models.passt import PaSST
>>> torch.manual_seed(0)
>>> batch_size, n_bins, n_frames = 4, 128, 50
>>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa")
>>> input = torch.randn((batch_size, n_bins, n_frames))
>>> output = model(input)
>>> print(output.size())
torch.Size([4, 527])
```

## License
- Apache License, Version 2.0 **EXCEPT FOR WEIGHTS OF PRETRAINED MODELS**
- Weights for some of the pre-trained models are extracted from the official implementations. Their licenses follow the official implementations.
