from credit_default.data import (
    BILL_AMOUNT_COLUMNS,
    CANONICAL_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    TARGET_COLUMN,
    load_credit_data,
)


def test_raw_dataset_contract():
    df = load_credit_data()

    assert df.shape == (30000, 25)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["ID"].is_unique
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}
    assert len(PAY_STATUS_COLUMNS) == 6
    assert len(BILL_AMOUNT_COLUMNS) == 6
    assert len(PAY_AMOUNT_COLUMNS) == 6
