import pandas as pd
import pycountry_convert as pc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "log_wages"

def run_pipeline_1(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Runs pipeline 1 - basic preprocessing. Handles only categorical column 'nationality_name'
    by string cleaning and grouping nationalities covering <10% of the data into 'other' category.
    Numerical features are standardised to 0 mean and variance=1 and cleaned 'nationality_name' is onehot encoded.
    Since no missing values are present in the data (found from inspection) no imputation is performed.

    Args:
        raw_df (pd.DataFrame): raw input dataframe

    Returns:
        X_processed (pd.DataFrame): preprocessed feature dataframe
        y (pd.Series): target variable series
        preprocessor (ColumnTransformer): fitted preprocessor object
    """
    df = raw_df.copy()

    if "nationality_name" in df.columns:
        df["nationality_name"] = (
            df["nationality_name"]
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
            .str.strip()
            .str.lower()
        )
        nat_counts = df["nationality_name"].value_counts(normalize=True).cumsum()
        COVERAGE = 0.90
        coverage_nats = nat_counts[nat_counts <= COVERAGE].index
        df["nationality_name"] = df["nationality_name"].apply(
            lambda x: x if x in coverage_nats else "other"
        )

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def run_pipeline_2(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Runs pipeline 2 - uses pycountry_convert to map nationalities to continents.
    Handles categorical column 'nationality_name' by mapping to continent codes.
    """
    df = raw_df.copy()

    if "nationality_name" in df.columns:
        # Clean basic artifacts first
        df["nationality_name"] = (
            df["nationality_name"]
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
            .str.strip()
            .str.title()  # Ensure Title Case for pycountry_convert
        )

        def get_continent(country):
            try:
                country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
                continent_code = pc.country_alpha2_to_continent_code(country_code)
                return continent_code
            except:
                # Handle specific cases common in football data
                if country.lower() in ['england', 'scotland', 'wales', 'northern ireland']:
                    return 'EU'
                return "other"

        df["continent"] = df["nationality_name"].apply(get_continent)
        df = df.drop(columns=["nationality_name"])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return pd.DataFrame(X_processed), y, preprocessor
