"""Model base classes and the MNIST, USPS, and SVHN classifiers."""

from abc import ABC, abstractmethod

from torch import nn


class BaseClassifier(nn.Module, ABC):
    """Abstract base: subclasses implement :meth:`forward(batch) -> logits` of shape ``(N, num_classes)``."""

    @abstractmethod
    def forward(self, x):
        """Return per-class logits for the input batch ``x``."""
        raise NotImplementedError


class DigitCNN(BaseClassifier, ABC):
    """Shared backbone for grayscale digit classifiers.

    Default 2-conv extractor + 128→10 head; subclasses override
    :meth:`_build_features` or :meth:`_build_classifier` to swap either piece.
    Abstract — use :class:`MNISTNet` or :class:`USPSNet`.

    Parameters
    ----------
    input_size : int, default 28
        Expected square spatial size; :meth:`forward` rejects mismatches.
    """

    def __init__(self, input_size: int = 28):
        """Build the shared backbone and classifier head sized for ``input_size``."""
        super().__init__()
        self.input_size = input_size
        self.feature_dim = input_size // 4

        self.features = self._build_features()
        self.classifier = self._build_classifier(in_features=64 * self.feature_dim**2)

    def _build_features(self) -> nn.Sequential:
        """Default feature extractor. 
        Subclasses can override this to change the CNN backbone structure.
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def _build_classifier(self, in_features: int) -> nn.Sequential:
        """Default classifier head.
        Subclasses can override this to change the classifier structure.
        """
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Run features → flatten → classifier on a ``(N, 1, input_size, input_size)`` batch; raises ``ValueError`` on shape mismatch."""
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Input size mismatch. Expected ({self.input_size}, {self.input_size}), "
                f"but got {x.shape[-2:]}."
            )
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class MNISTNet(DigitCNN):
    """CNN specialized for 28x28 MNIST digits."""

    def __init__(self, input_size: int):
        """Configure the shared :class:`DigitCNN` backbone.

        Parameters
        ----------
        input_size : int
            Square spatial size of incoming batches (typically 28 for MNIST).
        """
        # Inherit both the backbone and the head logic
        super().__init__(input_size=input_size)


class USPSNet(DigitCNN):
    """CNN specialized for 16x16 USPS digits."""

    def __init__(self, input_size: int):
        """Configure the shared :class:`DigitCNN` backbone.

        Parameters
        ----------
        input_size : int
            Square spatial size of incoming batches (typically 16 for USPS).
        """
        # Reuse the shared backbone and head; DigitCNN sizes the classifier
        # from the 64-channel backbone output (64 * feature_dim**2).
        super().__init__(input_size=input_size)

    def _build_features(self):
        """BatchNorm-augmented variant of the default 2-conv backbone."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def _build_classifier(self, in_features):
        """Dropout-regularized 2-layer head (128 hidden units, 10 outputs)."""
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )


class SVHNNet(BaseClassifier):
    """CNN specialized for 32x32 RGB SVHN digits."""

    def __init__(self, input_size: int, num_classes: int = 10, dropout: float = 0.3):
        """Configure the SVHN backbone and classifier head.

        Parameters
        ----------
        input_size : int
            Square spatial size of incoming batches (typically 32 for SVHN).
        num_classes : int, default 10
            Number of output classes.
        dropout : float, default 0.3
            Dropout probability used in the classifier head (not the feature blocks).
        """
        super().__init__()
        self.input_size = input_size
        feature_dim = input_size // 8

        def block(in_ch, out_ch):
            """Two Conv→BN→ReLU layers followed by 2×2 max pooling."""
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 32x32 -> 16x16
            block(32, 64),   # 16x16 -> 8x8
            block(64, 128),  # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * feature_dim**2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Run features → flatten → classifier on a ``(N, 3, input_size, input_size)`` batch; raises ``ValueError`` on shape mismatch."""
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Input size mismatch. SVHNNet expects ({self.input_size}, {self.input_size}), "
                f"but got {x.shape[-2:]}."
            )
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


__all__ = ["BaseClassifier", "DigitCNN", "MNISTNet", "USPSNet", "SVHNNet"]
