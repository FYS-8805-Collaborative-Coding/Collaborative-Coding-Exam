import pytest
import torch

from src.models import BaseClassifier, DigitCNN, MNISTNet, USPSNet, SVHNNet

# (model class, input channels, square input size) 
MODEL_SHAPES = [
    (MNISTNet, 1, 28),
    (USPSNet, 1, 16),
    (SVHNNet, 3, 32),
]


@pytest.mark.parametrize("model_cls, channels, size", MODEL_SHAPES)
@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_forward_output_shape(model_cls, channels, size, batch_size):
    """
    Test that the forward pass produces the expected output shape and dtype for
    a given batch size, input channels, and spatial dimensions.
    """
    model = model_cls(input_size=size).eval()
    logits = model(torch.randn(batch_size, channels, size, size))
    assert logits.shape == (batch_size, 10)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("model_cls, channels, size", MODEL_SHAPES)
def test_forward_not_hardcoded_batch(model_cls, channels, size):
    """
    Test that the forward pass does not have a hard-coded batch dimension, by
    passing a different batch size and checking that the output shape matches.
    """
    model = model_cls(input_size=size).eval()
    logits = model(torch.randn(3, channels, size, size))
    assert logits.shape[0] == 3


def test_svhnnet_custom_num_classes():
    """
    Test that SVHNNet can be instantiated with a custom number of output classes,
    and that the forward pass produces the expected output shape.
    """
    model = SVHNNet(input_size=32, num_classes=5).eval()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 5)


def test_concrete_models_subclass_baseclassifier():
    """
    Test that all concrete model classes are subclasses of BaseClassifier and can be
    instantiated without error.
    """
    for model_cls in (MNISTNet, USPSNet, SVHNNet):
        assert issubclass(model_cls, BaseClassifier)
        # Use a dummy size for checking instance type
        dummy_size = 32 if model_cls == SVHNNet else 28
        model = model_cls(input_size=dummy_size)
        assert isinstance(model, torch.nn.Module)


def test_baseclassifier_cannot_be_instantiated():
    """
    Test that the abstract base class BaseClassifier cannot be instantiated directly,
    and that it raises a TypeError when attempted.
    """
    with pytest.raises(TypeError):
        BaseClassifier()


@pytest.mark.parametrize("model_cls, channels, size", MODEL_SHAPES)
def test_wrong_channel_count_raises(model_cls, channels, size):
    """
    Test that passing an input tensor with the wrong number of channels raises a RuntimeError,
    which is the typical error for a shape mismatch in a convolutional layer.
    """
    model = model_cls(input_size=size).eval()
    wrong_channels = 3 if channels == 1 else 1
    with pytest.raises(RuntimeError):
        model(torch.randn(4, wrong_channels, size, size))


def test_wrong_spatial_size_raises():
    """
    Test that passing an input tensor with the wrong spatial dimensions raises a ValueError
    with a descriptive message.
    """
    model = MNISTNet(input_size=28).eval()
    with pytest.raises(ValueError, match="Input size mismatch"):
        model(torch.randn(4, 1, 32, 32)) 


def test_digitcnn_respects_input_size():
    """
    Test that DigitCNN can be instantiated with different input sizes and that the forward pass
    produces the expected output shape.
    """
    for size in (28, 16):
        model = DigitCNN(input_size=size).eval()
        logits = model(torch.randn(2, 1, size, size))
        assert logits.shape == (2, 10)
