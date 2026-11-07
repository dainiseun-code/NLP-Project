"""Pure encoder identity and exclusive, content-bound extraction artifacts.

No optional model runtime is imported here. Expected identities are supplied by
trusted code/sealed inputs, never inferred from the cache being checked.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from messaging_nlp.contracts import Digest, ExtractionResult, NonBlank, StrictModel
from messaging_nlp.io import canonical_json, read_json


class FileIdentity(StrictModel):
    bytes: int = Field(gt=0)
    sha256: Digest


class EncoderSettings(StrictModel):
    max_tokens: Literal[256] = 256
    threads: Literal[2] = 2
    interop_threads: Literal[2] = 2
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32"] = "float32"
    deterministic: Literal[True] = True
    attention: Literal["eager"] = "eager"
    truncation: Literal[False] = False

    @field_validator("max_tokens", "threads", "interop_threads", mode="before")
    @classmethod
    def integer_settings(cls, value):
        if type(value) is not int:
            raise ValueError("encoder counts require strict integers")
        return value

    @field_validator("deterministic", "truncation", mode="before")
    @classmethod
    def boolean_settings(cls, value):
        if type(value) is not bool:
            raise ValueError("encoder flags require strict booleans")
        return value


class EncoderIdentity(StrictModel):
    schema_version: Literal[1] = 1
    format: Literal["nlp3-xlmr-final-block-bio-head-v1"] = (
        "nlp3-xlmr-final-block-bio-head-v1"
    )
    repo: NonBlank
    revision: NonBlank
    base_directory: NonBlank
    base_files: dict[str, FileIdentity]
    receipt_path: NonBlank
    receipt_sha256: Digest
    component_directory: NonBlank
    component_config_sha256: Digest
    component_sha256: Digest
    hidden_size: int = Field(gt=0)
    final_block_index: int = Field(ge=0)
    labels: list[str]
    offset_policy: Literal["whitespace_edges_unowned_marks_v1"] = (
        "whitespace_edges_unowned_marks_v1"
    )
    decoder_policy: Literal["maximum_score_valid_BIO_path_no_gold"] = (
        "maximum_score_valid_BIO_path_no_gold"
    )
    settings: EncoderSettings = EncoderSettings()

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("identity schema requires a strict integer")
        return value

    @model_validator(mode="after")
    def valid_identity(self):
        from messaging_nlp.encoder_decode import LABELS

        for path in (self.base_directory, self.receipt_path, self.component_directory):
            portable_path(path)
        required = {
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        if set(self.base_files) != required or self.labels != list(LABELS):
            raise ValueError("unexpected encoder file set or label order")
        return self

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump())).hexdigest()


def portable_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or "\\" in value
        or not path.parts
        or path == PurePosixPath(".")
    ):
        raise ValueError("expected explicit portable project-relative path")
    return path


def retained_identity() -> EncoderIdentity:
    """Code-owned anchors, independent of mutable adjacent model metadata.

    These are the accepted copied-asset receipt/config identities. This function
    opens no assets; real byte verification belongs to the authorized loader.
    """
    files = {
        "config.json": (
            615,
            "d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579",
        ),
        "model.safetensors": (
            1115567652,
            "6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb",
        ),
        "sentencepiece.bpe.model": (
            5069051,
            "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        ),
        "tokenizer.json": (
            9096718,
            "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c",
        ),
        "tokenizer_config.json": (
            25,
            "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
        ),
    }
    from messaging_nlp.encoder_decode import LABELS

    return EncoderIdentity(
        repo="FacebookAI/xlm-roberta-base",
        revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        base_directory="models/xlm-roberta-base",
        base_files={
            name: FileIdentity(bytes=size, sha256=digest)
            for name, (size, digest) in files.items()
        },
        receipt_path="models/receipts/xlmr.json",
        receipt_sha256="819a41ab714228e4da9228444472352fea24a87bd8f1e2ad14cfd88971099955",
        component_directory="models/xlmr-final-block-head",
        component_config_sha256="1448aafb1814250f8efc49c3891e244342031fc5bc980a09a84f85558fb1886e",
        component_sha256="220cf11dff06468542706a6bcfc8f2ef12f3a99971e80364fd1fbdd46629fdee",
        hidden_size=768,
        final_block_index=11,
        labels=list(LABELS),
    )


class ExtractionArtifact(StrictModel):
    schema_version: Literal[1] = 1
    text: NonBlank
    identity: EncoderIdentity
    result: ExtractionResult
    diagnostics: dict

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema version requires a strict integer")
        return value

    @model_validator(mode="after")
    def validate_binding(self):
        if self.result.input_sha256 != text_sha256(self.text):
            raise ValueError("cache exact-text hash mismatch")
        if self.result.extractor_sha256 != self.identity.digest():
            raise ValueError("cache extractor identity hash mismatch")
        if self.result.extraction is not None:
            self.result.extraction.validate_text(self.text)
        canonical_json(self.diagnostics)
        return self


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_artifact(path: Path, artifact: ExtractionArtifact) -> None:
    """Exclusive fsynced write. Failed/partial prior artifacts remain untouched."""
    artifact = ExtractionArtifact.model_validate(artifact.model_dump())
    data = canonical_json(artifact.model_dump()) + b"\n"
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_artifact(
    path: Path, *, message_id: str, text: str, identity: EncoderIdentity
) -> ExtractionArtifact:
    """Validate a saved outcome, with no regeneration or model access."""
    artifact = ExtractionArtifact.model_validate(read_json(Path(path)))
    if artifact.text != text or artifact.result.message_id != message_id:
        raise ValueError("stale cache request identity or exact text")
    if canonical_json(artifact.identity.model_dump()) != canonical_json(
        identity.model_dump()
    ):
        raise ValueError("stale cache extractor identity")
    return artifact
