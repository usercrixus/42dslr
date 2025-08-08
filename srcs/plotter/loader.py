import pandas as pd


def load_and_prepare(path):
    """
    Load CSV, convert types for specific columns:
      - Birthday → epoch seconds (if present)
      - Best Hand → 1/0 (if present)
      - Hogwarts House → label-encoded (if present)
      - coerce all other columns to numeric
      - fill NaN with 0
    """
    df = pd.read_csv(path)

    # Convert Birthday column if it exists
    if "Birthday" in df.columns:
        df["Birthday"] = (
            pd.to_datetime(df["Birthday"], errors="coerce").astype("int64") // 10**9
        )

    # Map Best Hand if it exists
    if "Best Hand" in df.columns:
        df["Best Hand"] = df["Best Hand"].map({"Left": 1.0, "Right": 0.0})

    # Label encode Hogwarts House if present
    if "Hogwarts House" in df.columns:
        houses = sorted(df["Hogwarts House"].dropna().unique())
        mapping = {h: i for i, h in enumerate(houses)}
        df["Hogwarts House"] = df["Hogwarts House"].map(mapping)

    # Coerce any object-typed columns into numeric (invalid parsing → NaN)
    df = df.apply(
        lambda col: pd.to_numeric(col, errors="coerce") if col.dtype == object else col
    )

    # Replace all NaNs with 0
    df = df.drop(
        columns=["Index", "First Name", "Last Name", "Best Hand"], errors="ignore"
    )
    
    return df.fillna(0)
