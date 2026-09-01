# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def extract_boxed(text: str, *, last: bool = False) -> str | None:
    marker = "boxed{"
    marker_index = text.rfind(marker) if last else text.find(marker)
    if marker_index == -1:
        return None

    content_start = marker_index + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]

    return None
