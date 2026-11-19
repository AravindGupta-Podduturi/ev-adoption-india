# Data sources, extraction protocol and provenance

Panel: 4 states x Jan-2022..Oct-2026 (58 months, 232 rows). Registrations are used through Sep-2026 (Oct-2026 is a partial month and is dropped);
air quality, weather and fuel prices through Aug-2026 (Sep/Oct-2026 values in the original extract were identical across states, i.e. placeholders, and are set to NaN).
The extraction protocol below is the team's own "Master Data Extraction & Pipeline Manual" (project file *1st to do list*); the raw files follow its folder structure
(`data/raw/vahan`, `cpcb_aqi`, `ppac_fuel`, `nasa_weather`, `charging_stations`, `economic_nsdp`). Raw exports are not in this repository (size / licence); the model-ready
extract is `data/raw/ev_adoption_ml_dataset_original.csv`, and every transformation after it is scripted (`00a_apply_reference_data.py`, `00_build_vintage_panel.py`).

## 1. Source register

| Column group | Source and exact specification | Granularity | Alignment / release rule in the vintage panel |
|---|---|---|---|
| `vehicle_total`, `ev_count`, fuel and category splits | MoRTH **Vahan public dashboard** (analytics.parivahan.gov.in/analytics/publicdashboard/vahan), **downloaded 9 Oct 2026**. Filters: State = Delhi, Maharashtra, Tamil Nadu or West Bengal; X-axis = *Month Wise*; Y-axis = *Fuel* (one export) and *Vehicle Category* (second export); years 2022..2026, one Excel file per state-year-view. Raw category columns: four-wheeler (invalid carriage), heavy/medium/light goods, heavy/medium/light motor, heavy/medium/light passenger, other, three-wheeler (invalid carriage / NT / T), two-wheeler (invalid carriage / NT / T). Raw fuel columns: 34 fuel codes (petrol and E20 variants, diesel, CNG, LPG, hybrids, ELECTRIC(BOV), PURE EV, plug-in and strong hybrid, hydrogen, solar, etc.). Both extracts were later combined into the grouped columns of the model dataset (`two_wheeler`, `three_wheeler`, `four_wheeler`, `light/medium/heavy_vehicle`, `other`; `petrol`, `diesel`, `gas`, `electric`, `hybrid`, `lpg`, `alternative`, `not_applicable`). EV = the `electric` group (battery-electric); in the dataset `ev_count` equals `electric` exactly and hybrids are a separate group; both extracts sum to `vehicle_total` in all 232 rows. The exact raw-to-group mapping was not recorded as a file (see section 6). Share = EV / total registrations | state-month | target; not used as predictor. Vahan numbers are revised retrospectively; September 2026 was downloaded nine days after month end and may be slightly incomplete |
| `aqi_mean`, `aqi_max`, `pm25_mean`, `pm10_mean`, `no2_mean`, `so2_mean`, `dominant_pollutant` | **CPCB CAAQMS dashboard** (app.cpcbccr.com/ccr), *Advanced search / historical data*, criteria *daily average*, **downloaded 9 Oct 2026**; parameters PM2.5, PM10, NO2, SO2, AQI. **One station per state**: Delhi = **RK Puram**; Maharashtra = Mumbai - Bandra; Tamil Nadu = Chennai - Alandur; West Bengal = Kolkata - Victoria. Annual query windows, one CSV per station-year | monthly mean of the daily station values | usable one month later. State-wide air quality is NOT measured: the station is a proxy for the state's main metro |
| `mean_temp_c`, `precipitation_mm` | **NASA POWER** point API (power.larc.nasa.gov), variables T2M (2 m temperature, C) and PRECTOTCORR (corrected precipitation). **Four representative cities per state, arithmetic mean across them**: Delhi = Delhi, Gurugram, Noida, Faridabad; Maharashtra = Mumbai, Pune, Nagpur, Nashik; Tamil Nadu = Chennai, Coimbatore, Madurai, Tiruchirappalli; West Bengal = Kolkata, Asansol, Siliguri, Durgapur (coordinates in `scripts/download_nasa_power_weather_2026_jan_aug.py`). The 2026 months were retrieved with the daily endpoint and aggregated (T2M = mean of valid daily values, precipitation = sum of valid daily values in mm/month, -999 = missing); 2022-2025 were extracted the same way per the team | monthly | usable one month later. Weather is a four-city average whereas air quality is one station, and Delhi's cities include Gurugram, Noida and Faridabad (outside the Delhi territory); Siliguri has a different climate from Kolkata |
| `petrol_price`, `diesel_price` | **PPAC** "Daily Metro Retail Selling Prices" (ppac.gov.in, Prices > Historical Retail Prices): Delhi, Mumbai, Chennai, Kolkata. Verified in the file: June-2025 petrol = 94.77 (Delhi), 103.50 (Mumbai), 100.80 (Chennai), 105.41 (Kolkata) | daily -> monthly mean | usable one month later |
| `charging_stations` | Parliamentary replies and PIB releases, compiled in `data/external/charging_snapshots.csv` (4 snapshots per state; all-India totals in `charging_all_india_context.csv`, context only). See section 2 | snapshot | explicit assumed release month per snapshot; never back-filled; Jan-2022..Feb-2023 take the first snapshot and are flagged `charging_backfilled=1` |
| `nsdp_per_capita` (**per-capita GSDP**, gross, current prices) | `data/external/gsdp_per_capita_fy.csv`: financial years 2022-23 (Actual), 2023-24 (Revised), 2024-25 (Advance), 2025-26 (Advance), 2026-27 (Projected, derived). Gross state domestic product per capita is used as supplied; the column name `nsdp_per_capita` is kept for code compatibility. Source editions are in section 3 | financial year (Apr-Mar) | usable 24 months after the FY starts (FY2022-23 from Apr-2024); the projected FY2026-27 value never reaches a forecast. Jan-Mar 2022 (FY2021-22 not supplied) take FY2022-23 and are flagged `nsdp_backfilled=1` |

## 2. Charging-station snapshots: what each number is
| Snapshot | Source document (verified) | Definition used by the source | Values used (MH / DL / TN / WB) |
|---|---|---|---|
| Mar-2023 | PIB, Ministry of Heavy Industries, 24 Mar 2023 (data as of 21 Mar 2023), Annexure A. All-India 6,586 | **operational** public charging stations (data from the Bureau of Energy Efficiency) | 660 / 1,845 / 441 / 189 |
| Feb-2024 | PIB, Ministry of Heavy Industries, 6 Feb 2024 (data as of 2 Feb 2024, Ministry of Power inputs), Annexure II. All-India 12,146 | **operational** public charging stations | 3,079 / 1,886 / 643 / 318 |
| Dec-2025 | PIB, Ministry of Heavy Industries, 16 Dec 2025 (Lok Sabha written reply). All-India 29,151 | stations **installed in the last five years** (fast + slow); *not* necessarily operational | 4,166 / 1,957 / 1,780 / 891 |
| Aug-2026 | Rajya Sabha Unstarred Question No. 2333, *Installation of Electric Vehicle Charging Stations*, Ministry of Heavy Industries, answered 7 Aug 2026 (annexure AU2333_V2LOXS.pdf on sansad.in). The all-India total 52,718 was first cited in a Lok Sabha written reply of 21 Jul 2026 (also reported by IANS) | currently operational (per press report) | 6,985 / ~3,800* / 3,839 / ~1,200* |

*Asterisk (*): the Delhi and West Bengal Aug-2026 values are estimates, not verbatim readouts; published summaries of the 7 Aug 2026 annexure reported exact counts only for the top states (Maharashtra 6,985, Tamil Nadu 3,839). Replace them with the annexure values when the PDF is checked.*

**Definition break.** The first two snapshots count operational stations, Dec-2025 counts installations over five years, Aug-2026 counts operational stations again.
The series is therefore not like-for-like. It is treated as a coarse level indicator, and results never rely on its month-to-month change. Corrections made to the
team's first table are in `data/external/CORRECTIONS_LOG.md` (Maharashtra Feb-2024 3,728 -> 3,079; the unsourced "~end-2022" row replaced by the official Mar-2023 PIB row).

## 3. Per-capita GSDP
Values (INR, current prices) are in `data/external/gsdp_per_capita_fy.csv` with a status column. Source tiers by financial year:

| Financial year | Status | Source editions |
|---|---|---|
| 2022-23, 2023-24 | Actual / first revised estimate | MoSPI State Domestic Product series (e-Sankhyiki portal, release of 17 Apr 2026) and the state Directorates of Economics and Statistics / Economic Surveys (2024-25 and 2025-26 editions) |
| 2024-25, 2025-26 | Advance estimates | MoSPI e-Sankhyiki release (17 Apr 2026), reconciled with the 2025-26 state Economic Surveys (Maharashtra, published 5 Mar 2026; Delhi; Tamil Nadu; West Bengal Economic Review / budget publications) |
| 2026-27 | Projected (not MoSPI) | 2026-27 state budget documents: each state's budgeted nominal GSDP growth applied to the 2025-26 baseline; derived estimates, not official actuals |

Base-year note: MoSPI moved national accounts to the new series (base 2022-23) on 27 Feb 2026; state releases may still carry back-series on the older 2011-12 base, and price bases differ across state releases. The variable is therefore used only as a slow-moving income level. The MoSPI table cited reports per-capita *net* state domestic product; the series used here is *gross* per capita as supplied by the team and was not changed (limitation). It never enters a forecast before the 24-month release rule allows, so the projected 2026-27 value is never visible inside the data window.

Full references (APA 7th): Directorate of Economics and Statistics, Government of Maharashtra (2026, March 5), *Economic survey of Maharashtra 2025-26*; Finance Department, Government of West Bengal (2026), *Economic review and budget publications 2025-26 / 2026-27*; Ministry of Heavy Industries (2026, August 7), *Installation of electric vehicle charging stations* [Rajya Sabha Unstarred Question No. 2333]; MoSPI (2026, April 17), *National accounts statistics: state domestic product* [data set], e-Sankhyiki portal; Planning Department, Government of NCT of Delhi (2026, March), *Economic survey of Delhi 2025-26*; State Planning Commission, Government of Tamil Nadu (2026, February), *Economic survey of Tamil Nadu 2025-26*. URLs are listed in `report/refs.tex` notes.

## 4. Release (vintage) rules - assumptions, stated explicitly
* Monthly covariates (air quality, weather, fuel): visible one month after the month ends.
* Charging snapshots: published label+1 month (ministry/parliament replies), never back-filled.
* GSDP: visible 24 months after the financial year begins (conservative: RBI's Handbook edition covering 2024-25 appeared in December 2025, <https://currentaffairs.bankexamstoday.com/2025/12/rbi-releases-10th-edition-of-handbook.html>).
* Change `config/pipeline_config.json` / `data/external/*.csv` to test other lags.

## 5. West Bengal Nov-2025..Feb-2026 episode
The surge coincides with a state mandate to register e-rickshaws (Oct-2025; deadline 31-Dec-2025 extended to 31-Jan-2026) reported by trade press
(<https://emobilityplus.com/2026/02/09/west-bengal-e-rickshaw-drive-uncovers-hidden-ev-fleet-in-india>). In the panel, excess EVs / excess three-wheelers = 1.01 (correlation 0.98), and the other
states show no such jump. This is **evidence consistent with an e-rickshaw registration backlog; it is not an official verification**, and the "backlog-removed" series is an
assumption-based sensitivity treatment, not a corrected ground-truth dataset.

## 6. Provenance status and remaining checks
| # | Item | Status |
|---|---|---|
| 1 | Vahan and CPCB download date | 9 Oct 2026 (recorded above) |
| 2 | NASA POWER points and script | Four cities per state; script in `scripts/` (run date is printed in the output file's Metadata sheet) |
| 3 | Delhi air-quality station | RK Puram |
| 4 | Aug-2026 charging reply | Rajya Sabha Unstarred Q.2333, 7 Aug 2026. Delhi and West Bengal values are estimates; check against the annexure PDF |
| 5 | GSDP source editions | Section 3 |
| 6 | Raw-to-group mapping of Vahan columns | Not stored as a file. Category groups follow the column names; hybrids (including plug-in and strong hybrid) are assumed to be in `hybrid`, and `electric` is the battery-electric group. Correct here if different |
| 7 | GSDP gross vs net | Gross used as supplied; the MoSPI table lists net per capita (limitation) |
