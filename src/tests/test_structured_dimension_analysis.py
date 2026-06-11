from src.analysis.structured_dimension_analysis import normalize_company_size, normalize_experience


def test_normalize_experience_buckets_common_values():
    assert normalize_experience("经验不限") == "no_experience_required"
    assert normalize_experience("1年经验") == "0_1_year"
    assert normalize_experience("3-5年") == "1_3_years"
    assert normalize_experience("10年以上") == "5_10_years"


def test_normalize_company_size_buckets_common_values():
    assert normalize_company_size("少于20人") == "lt_20"
    assert normalize_company_size("20-99人") == "20_99"
    assert normalize_company_size("100-499人") == "100_499"
    assert normalize_company_size("10000人以上") == "10000_plus"
