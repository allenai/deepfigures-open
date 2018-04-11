"""Exceptions for deepfigures."""


class LatexException(OSError):
    """An exception thrown for errors in rendering LaTeX."""

    def __init__(self, cmd, code, stdout):
        self.code = code
        self.stdout = stdout

    def __str__(self):
        return (
            'Return code: %s, stdout: %s' %
            (repr(self.code), repr(self.stdout))
        )


class PDFProcessingError(OSError):
    """An exception thrown for errors in processsing a PDF."""
