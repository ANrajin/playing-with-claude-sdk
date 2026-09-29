from __future__ import annotations

from typing import Any


def calculate_average(numbers: list[int | float]) -> float:
    if not numbers:
        raise ValueError("Cannot calculate average of an empty or null sequence")
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def get_user_name(user: dict[str, Any] | None) -> str:
    if user is None:
        raise ValueError("user cannot be None")
    name = user.get("name")
    if name is None:
        raise KeyError("user dict is missing the 'name' key or its value is None")
    return name.upper()