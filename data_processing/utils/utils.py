import pandas as pd
import datetime as dt


def load_data(file_path):
    """
    Load data from an excel file
    """

    return pd.read_excel(file_path)


def treat_missing_values(df):
    """
    Treat missing values in the dataframe
    """

    # if driver_other_vehicles is missing, fill with "NONE"
    df["driver_other_vehicles"] = df["driver_other_vehicles"].fillna("NONE")
    # if driver insured years is missing, fill with "ZERO"
    df["driver_insured_years"] = df["driver_insured_years"].fillna("ZERO")

    # Create a new column 'has_occasional_driver' based on occasional_driver_birth_date.
    # If the value is missing, fill with "False" else "True"
    df["has_occasional_driver"] = df["occasional_driver_birth_date"].notnull()

    # Drop 'occasional_driver_birth_date' column
    df = df.drop("occasional_driver_birth_date", axis=1)
    # Drop 'occasional_driver_license_attainment_age' column
    df = df.drop("occasional_driver_license_attainment_age", axis=1)

    # Create a new column 'has_other_policyholder' based on policyholder_age.
    # If the value is missing, fill with "False" else "True"
    df["has_other_policyholder"] = df["policyholder_age"].notnull()

    # Drop 'policyholder_age' column
    df = df.drop("policyholder_age", axis=1)
    # Drop 'policyholder_license_attainment_age' column
    df = df.drop("policyholder_license_attainment_age", axis=1)

    return df


def clean_column_types(df):
    """
    Cast columns to the correct data types
    """

    # Cast columns in year-month format to datetime
    df["vehicle_registration_ym"] = pd.to_datetime(
        df["vehicle_registration_ym"], format="%Y-%m"
    )
    df["vehicle_buy_ym"] = pd.to_datetime(df["vehicle_buy_ym"], format="%Y-%m")
    df["driver_driving_license_ym"] = pd.to_datetime(
        df["driver_driving_license_ym"], format="%Y-%m"
    )

    return df


def encode_categorical_features(df):
    """
    Encode categorical features
    """

    # Do one-hot encoding for driver_other_vehicles and vehicle_acquisition_state
    df = pd.get_dummies(
        df, columns=["driver_other_vehicles", "vehicle_acquisition_state"]
    )
    # Do ordinal encoding for driver_insured_years
    df["driver_insured_years"] = df["driver_insured_years"].map(
        {
            "ZERO": 0,
            "ONE": 1,
            "TWO": 2,
            "THREE": 3,
            "FOUR": 4,
            "FIVE": 5,
            "SIX": 6,
            "SEVEN": 7,
            "EIGHT": 8,
            "NINE": 9,
            "TEN": 10,
            "MORE_THAN_TEN": 11,
        }
    )
    # Create a column professional_vehicle_use if vehicle_use is 'PROFESSIONAL_HABITUAL' OR 'PROFESSIONAL_OCCASIONAL'
    df["professional_vehicle_use"] = df["vehicle_use"].apply(
        lambda x: x in ["PROFESSIONAL_HABITUAL", "PROFESSIONAL_OCCASIONAL"]
    )
    # Create a column habitual_vehicle_use if vehicle_use is 'PROFESSIONAL_HABITUAL' OR 'HABITUAL'
    df["habitual_vehicle_use"] = df["vehicle_use"].apply(
        lambda x: x in ["PROFESSIONAL_HABITUAL", "HABITUAL"]
    )
    # Drop 'vehicle_use' column
    df = df.drop("vehicle_use", axis=1)

    return df


def generate_features(df):
    """
    Generate new features
    """

    # Create column driver_age based on driver_birth_date
    df["driver_age"] = (dt.datetime.now() - df["driver_birth_date"]).dt.days // 365
    # Drop 'driver_birth_date' column
    df = df.drop("driver_birth_date", axis=1)
    # Create years_with_license based on driver_driving_license_ym
    df["years_with_license"] = (
        dt.datetime.now() - df["driver_driving_license_ym"]
    ).dt.days // 365
    # Drop 'driver_driving_license_ym' column
    df = df.drop("driver_driving_license_ym", axis=1)
    # Create driver_has_other_vehicles based on driver_other_vehicles
    df["driver_has_other_vehicles"] = df["driver_other_vehicles_NONE"] == 0
    # Create driver_has_been_insurant based on driver_insured_years
    df["driver_has_been_insured"] = df["driver_insured_years"] > 0
    # Create years_since_bought based on vehicle_buy_ym
    df["years_since_bought"] = (dt.datetime.now() - df["vehicle_buy_ym"]).dt.days // 365
    # Drop 'vehicle_buy_ym' column
    df = df.drop("vehicle_buy_ym", axis=1)
    # Create years_since_registered based on vehicle_registration_ym
    df["years_since_registered"] = (
        dt.datetime.now() - df["vehicle_registration_ym"]
    ).dt.days // 365
    # Drop 'vehicle_registration_ym' column
    df = df.drop("vehicle_registration_ym", axis=1)
    # Create driver_claims summing all driver_claims columns
    df["driver_claims"] = (
        df["driver_claims_from_year_1_to_2"]
        + df["driver_claims_from_year_2_to_3"]
        + df["driver_claims_from_year_3_to_4"]
        + df["driver_claims_from_year_4_to_5"]
    )

    return df


def save_data(df, file_path):
    """
    Save data to an excel file
    """

    df.to_excel(file_path, index=False)
