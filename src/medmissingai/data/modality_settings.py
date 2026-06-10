from __future__ import annotations

from itertools import combinations


DEFAULT_MODALITIES = ("t1", "t1ce", "t2", "flair")


def all_nonempty_settings(
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
) -> list[tuple[str, ...]]:
    settings: list[tuple[str, ...]] = []
    for size in range(1, len(modalities) + 1):
        settings.extend(combinations(modalities, size))
    return [tuple(setting) for setting in settings]


def setting_id(setting: tuple[str, ...]) -> str:
    return "+".join(setting)


def parse_settings(
    value: str,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
) -> list[tuple[str, ...]]:
    """Parse modality availability settings.

    Accepted values:
      all: all 15 non-empty subsets for four modalities.
      full: all modalities available.
      t1+t2,t1ce+flair: explicit comma-separated settings.
    """

    normalized = value.strip().lower()
    if normalized == "all":
        return all_nonempty_settings(modalities)
    if normalized == "full":
        return [modalities]

    known = set(modalities)
    parsed: list[tuple[str, ...]] = []
    for item in normalized.split(","):
        setting = tuple(part.strip() for part in item.split("+") if part.strip())
        if not setting:
            continue
        unknown = sorted(set(setting).difference(known))
        if unknown:
            raise ValueError(f"Unknown modality names in setting {item!r}: {unknown}")
        parsed.append(setting)

    if not parsed:
        raise ValueError("No modality settings were parsed")
    return parsed

