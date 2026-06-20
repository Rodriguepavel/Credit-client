from credit_default.data import TARGET_COLUMN, load_credit_data
from credit_default.preprocessing import (
    MODELING_BASE_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    build_modeling_base,
    clean_credit_default_data,
)


def test_cleaning_preserves_rows_and_target_rate():
    raw = load_credit_data()
    clean = clean_credit_default_data(raw)

    assert len(clean) == len(raw)
    assert clean[TARGET_COLUMN].mean() == raw[TARGET_COLUMN].mean()
    assert clean["education_group"].isna().sum() == 0
    assert clean["marriage_group"].isna().sum() == 0
    assert set(clean["education_group"].unique()) == {
        "graduate_school",
        "university",
        "high_school",
        "other_unknown",
    }


def test_modeling_base_has_expected_columns_and_no_missing_values():
    raw = load_credit_data()
    modeling = build_modeling_base(raw)

    assert list(modeling.columns) == MODELING_BASE_COLUMNS
    assert modeling.shape[0] == 30000
    assert modeling[MODEL_NUMERIC_COLUMNS].isna().sum().sum() == 0
    assert modeling[MODEL_CATEGORICAL_COLUMNS].isna().sum().sum() == 0
    assert set(modeling[TARGET_COLUMN].unique()) == {0, 1}
