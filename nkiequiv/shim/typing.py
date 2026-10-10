"""Shim for ``nki.typing`` (type annotations only)."""


class _Annotation:
    def __getitem__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return "nki.typing.tensor"


tensor = _Annotation()
