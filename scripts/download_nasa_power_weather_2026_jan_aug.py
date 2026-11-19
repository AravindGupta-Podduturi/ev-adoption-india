import requests
import pandas as pd
import calendar
from datetime import date
from pathlib import Path
from openpyxl import load_workbook

# ============================================================
# NASA POWER WEATHER DATA DOWNLOADER
# January-August 2026 | Delhi, Maharashtra, Tamil Nadu, West Bengal
#
# Uses the DAILY API because 2026 monthly meteorological data
# may not yet be available through the monthly endpoint.
#
# Parameters:
#   T2M         = Daily mean temperature at 2 m (°C)
#   PRECTOTCORR = Corrected daily precipitation (mm/day)
#
# Monthly aggregation:
#   T2M: mean of valid daily values
#   PRECTOTCORR: sum of valid daily values = monthly precipitation (mm)
#
# NASA POWER missing value -999 is treated as missing.
# ============================================================

START_YEAR = 2026
START_MONTH = 1
START_DAY = 1
END_YEAR = 2026
END_MONTH = 8
END_DAY = 31

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OUTPUT_FILE = "nasa_power_weather_2026_Jan_Aug.xlsx"

LOCATIONS = {
    "Delhi": {
        "Delhi": (28.6139, 77.2090),
        "Gurugram": (28.4595, 77.0266),
        "Noida": (28.5355, 77.3910),
        "Faridabad": (28.4089, 77.3178),
    },
    "Maharashtra": {
        "Mumbai": (19.0760, 72.8777),
        "Pune": (18.5204, 73.8567),
        "Nagpur": (21.1458, 79.0882),
        "Nashik": (19.9975, 73.7898),
    },
    "Tamil Nadu": {
        "Chennai": (13.0827, 80.2707),
        "Coimbatore": (11.0168, 76.9558),
        "Madurai": (9.9252, 78.1198),
        "Tiruchirappalli": (10.7905, 78.7047),
    },
    "West Bengal": {
        "Kolkata": (22.5726, 88.3639),
        "Asansol": (23.6739, 87.1510),
        "Siliguri": (26.7271, 88.3953),
        "Durgapur": (23.5204, 87.3119),
    },
}


def download_power_data(latitude, longitude):
    """Download daily NASA POWER data for one location."""
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
        "start": f"{START_YEAR}{START_MONTH:02d}{START_DAY:02d}",
        "end": f"{END_YEAR}{END_MONTH:02d}{END_DAY:02d}",
    }

    response = requests.get(API_URL, params=params, timeout=120)

    # Include NASA's error body so API validation messages are visible.
    if not response.ok:
        raise RuntimeError(
            f"NASA POWER API returned HTTP {response.status_code}. "
            f"Response: {response.text[:2000]}"
        )

    data = response.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        raise RuntimeError(
            "Unexpected NASA POWER response:\n" + str(data)[:2000]
        )
    return data


def parse_power_response(data, state, location, latitude, longitude):
    """Aggregate daily NASA POWER data into one row per month."""
    parameter_data = data["properties"]["parameter"]
    t2m = parameter_data.get("T2M", {})
    precipitation = parameter_data.get("PRECTOTCORR", {})

    records = []
    for period in t2m:
        # Daily API keys should be YYYYMMDD.
        if len(period) != 8 or not period.isdigit():
            continue

        year = int(period[:4])
        month = int(period[4:6])
        day = int(period[6:8])

        if (year, month, day) < (START_YEAR, START_MONTH, START_DAY):
            continue
        if (year, month, day) > (END_YEAR, END_MONTH, END_DAY):
            continue

        temp = t2m.get(period)
        precip_daily = precipitation.get(period)

        if temp == -999:
            temp = None
        if precip_daily == -999:
            precip_daily = None

        records.append({
            "State": state,
            "Location": location,
            "Latitude": latitude,
            "Longitude": longitude,
            "Year": year,
            "Month": month,
            "T2M_C": temp,
            "PRECTOTCORR_mm_day": precip_daily,
        })

    daily_df = pd.DataFrame(records)
    if daily_df.empty:
        return pd.DataFrame()

    # Aggregate daily observations to monthly values.
    monthly_df = (
        daily_df.groupby(
            ["State", "Location", "Latitude", "Longitude", "Year", "Month"],
            as_index=False
        )
        .agg(
            T2M_C=("T2M_C", "mean"),
            Precipitation_mm_month=("PRECTOTCORR_mm_day", "sum"),
            Valid_Temperature_Days=("T2M_C", "count"),
            Valid_Precipitation_Days=("PRECTOTCORR_mm_day", "count"),
        )
    )

    # Keep the existing daily precipitation column as a monthly daily average.
    monthly_df["PRECTOTCORR_mm_day"] = (
        monthly_df["Precipitation_mm_month"]
        / monthly_df["Valid_Precipitation_Days"].replace(0, pd.NA)
    )
    monthly_df.loc[
        monthly_df["Valid_Precipitation_Days"] == 0,
        "Precipitation_mm_month"
    ] = None

    # Keep the original main data columns and append valid-day counts.
    return monthly_df


# ============================================================
# DOWNLOAD ALL LOCATIONS
# ============================================================

all_data = []

print("=" * 70)
print("NASA POWER WEATHER DATA DOWNLOAD")
print("=" * 70)
print(f"Period: {START_YEAR}-{START_MONTH:02d}-{START_DAY:02d} "
      f"to {END_YEAR}-{END_MONTH:02d}-{END_DAY:02d}")
print("API: Daily")
print("Parameters: T2M, PRECTOTCORR")
print()

for state, locations in LOCATIONS.items():
    print(f"\nSTATE: {state}")
    print("-" * 50)

    for location, (latitude, longitude) in locations.items():
        print(f"Downloading {location}: lat={latitude}, lon={longitude}")
        try:
            data = download_power_data(latitude, longitude)
            df = parse_power_response(
                data, state, location, latitude, longitude
            )

            expected_months = 8
            if len(df) != expected_months:
                print(
                    f"  WARNING: received {len(df)} monthly rows; "
                    f"expected {expected_months}. Some dates may be unavailable."
                )

            if df.empty:
                print("  ERROR: no monthly records returned")
                continue

            all_data.append(df)
            print(f"  SUCCESS: {len(df)} monthly records")

        except requests.exceptions.RequestException as e:
            print(f"  ERROR: network/API request failed: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")


if not all_data:
    raise RuntimeError(
        "\nNo NASA POWER data were downloaded. "
        "Check the detailed API error above, your internet connection, "
        "and whether the requested data are available."
    )

raw_df = pd.concat(all_data, ignore_index=True)
raw_df = raw_df.sort_values(
    ["State", "Location", "Year", "Month"]
).reset_index(drop=True)

# State-level monthly aggregation across representative locations.
state_monthly = (
    raw_df.groupby(["State", "Year", "Month"], as_index=False)
    .agg(
        Mean_T2M_C=("T2M_C", "mean"),
        Mean_Precipitation_mm_month=("Precipitation_mm_month", "mean"),
        Valid_Temperature_Points=("T2M_C", "count"),
        Valid_Precipitation_Points=("Precipitation_mm_month", "count"),
    )
)
state_monthly["Month_Name"] = state_monthly["Month"].apply(
    lambda x: calendar.month_name[int(x)]
)
state_monthly = state_monthly[
    [
        "State", "Year", "Month", "Month_Name", "Mean_T2M_C",
        "Mean_Precipitation_mm_month", "Valid_Temperature_Points",
        "Valid_Precipitation_Points",
    ]
].sort_values(["State", "Year", "Month"]).reset_index(drop=True)

# Partial-period summary: averages of monthly temperatures and sums of
# state-level monthly precipitation for January-August 2026.
period_summary = (
    state_monthly.groupby(["State", "Year"], as_index=False)
    .agg(
        Annual_Mean_T2M_C=("Mean_T2M_C", "mean"),
        Annual_Precipitation_mm=("Mean_Precipitation_mm_month", "sum"),
    )
    .sort_values(["State", "Year"])
    .reset_index(drop=True)
)

coordinate_records = [
    {
        "State": state,
        "Location": location,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    for state, locations in LOCATIONS.items()
    for location, (latitude, longitude) in locations.items()
]
coordinates_df = pd.DataFrame(coordinate_records)

metadata = pd.DataFrame({
    "Field": [
        "Data source", "NASA POWER website", "API endpoint",
        "API documentation", "Start date", "End date", "Parameters",
        "T2M", "PRECTOTCORR", "Missing value handling",
        "Monthly temperature aggregation", "Monthly precipitation aggregation",
        "State-level aggregation", "Number of representative points per state",
        "States", "Access date",
    ],
    "Value": [
        "NASA Prediction Of Worldwide Energy Resources (POWER)",
        "https://power.larc.nasa.gov/",
        API_URL,
        "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
        f"{START_YEAR}-{START_MONTH:02d}-{START_DAY:02d}",
        f"{END_YEAR}-{END_MONTH:02d}-{END_DAY:02d}",
        "T2M, PRECTOTCORR",
        "Daily mean temperature at 2 meters, degrees Celsius",
        "Corrected daily precipitation, mm/day",
        "-999 treated as missing",
        "Mean of valid daily T2M values",
        "Sum of valid daily PRECTOTCORR values (mm/month)",
        "Arithmetic mean across up to four representative points",
        4,
        "Delhi; Maharashtra; Tamil Nadu; West Bengal",
        str(date.today()),
    ],
})

print()
print("=" * 70)
print("CREATING EXCEL FILE")
print("=" * 70)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    raw_df.to_excel(writer, sheet_name="Raw_Point_Data", index=False)
    state_monthly.to_excel(writer, sheet_name="State_Monthly", index=False)
    period_summary.to_excel(writer, sheet_name="State_Period_Summary", index=False)
    coordinates_df.to_excel(writer, sheet_name="Coordinates", index=False)
    metadata.to_excel(writer, sheet_name="Metadata", index=False)

wb = load_workbook(OUTPUT_FILE)
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = cell.font.copy(bold=True)
    for column in ws.columns:
        max_length = max(
            (len(str(cell.value)) for cell in column if cell.value is not None),
            default=0
        )
        ws.column_dimensions[column[0].column_letter].width = min(max_length + 2, 45)

for ws_name in ["Raw_Point_Data", "State_Monthly", "State_Period_Summary"]:
    ws = wb[ws_name]
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.00"

wb.save(OUTPUT_FILE)

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Output file: {Path(OUTPUT_FILE).resolve()}")
print(f"Raw location-month records: {len(raw_df)} (expected up to 128)")
print(f"State-month records: {len(state_monthly)} (expected up to 32)")
print(f"Period summary records: {len(period_summary)} (expected up to 4)")
print("Sheets: Raw_Point_Data, State_Monthly, State_Period_Summary, Coordinates, Metadata")
print("Note: Verify valid-day counts/available months before analysis.")
print("Done.")
