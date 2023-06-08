from __future__ import annotations

import numpy as np


class CategoricalValsValidator:
    @classmethod
    def must_be_in_existing_values(cls, value, unique_vals):
        if value not in unique_vals + [np.nan, None, 'nan']:
            raise ValueError(
                f'Unavailable {value}. Value must be in {unique_vals}',
            )
        return value
