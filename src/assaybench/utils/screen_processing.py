
from typing import List


def _extract_screen_ids_from_dataset_name(dataset_name: str) -> List[int]:
    """Extract one or more screen IDs from a dataset_name."""
    screen_ids: List[int] = []

    if dataset_name.startswith("U_TR_"):
        parts = dataset_name[5:].split("_")
        for part in parts[:-1]:
            try:
                screen_ids.append(int(part))
            except ValueError:
                continue
        return screen_ids

    if dataset_name.startswith("TR_"):
        parts = dataset_name[3:].split("_")
        for part in parts:
            try:
                screen_ids.append(int(part))
            except ValueError:
                continue
        return screen_ids

    if dataset_name.startswith("U_"):
        parts = dataset_name.split("_")
        if len(parts) >= 2:
            try:
                screen_ids.append(int(parts[1]))
            except ValueError:
                pass
        return screen_ids

    try:
        screen_ids.append(int(dataset_name))
    except ValueError:
        pass

    return screen_ids