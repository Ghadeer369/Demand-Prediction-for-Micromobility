# Dashboard Blueprint — Tableau Public

Built for Tableau Public because it runs natively on macOS (Power BI
Desktop is Windows-only, and Power BI Service in the browser is far more
limited). This matches the tool used in the original senior project too.

## 1. Install Tableau Public

Download from https://public.tableau.com/en-us/s/download (free, no account
needed to install — you'll create a free Tableau Public account to publish).

## 2. Connect to the data

Open Tableau Public → **Connect → Microsoft Excel** → select
`dashboard/Dashboard_Data.xlsx`. It has 4 sheets, each already formatted as
a proper Excel Table:

| Sheet | Grain | Use for |
|---|---|---|
| `Daily_Demand` | 1 row per day | Trends, KPIs, weekday/monthly charts |
| `Demand_By_Region` | 1 row per day × region | Zone Analysis page |
| `Model_Comparison` | 1 row per model | Model Comparison page |
| `Future_Predictions` | 1 row per future day | Predictions page |

Drag each sheet onto the canvas as a separate data source (they don't need
to be joined — each dashboard page below uses one sheet).

## 3. Brand colors — set these as a custom palette

Match the Spiders Mobility presentation deck so the dashboard, the charts
in `outputs/figures/`, and the README all look like one product:

- **Primary (mint green)**: `#4CAF7D`
- **Accent (lavender-blue, for weekends/highlights)**: `#7B85C9`
- **Text (charcoal)**: `#2D2D2D`

In Tableau: Format → Workbook Theme, or set it per-chart via the Color
shelf → Edit Colors → Custom Diverging/Sequential using the hex codes above.

## 4. Suggested sheets/pages

1. **Executive Summary** — KPI text tiles (total rides, avg daily rides,
   peak day, % weekend share) + a line chart of `Ride_Count` over
   `Ride_Start_Date` from `Daily_Demand`.
2. **Demand Trends** — bar chart of average `Ride_Count` by `Day_of_Week`
   (color `Is_Weekend` with the two brand colors), bar chart by month.
3. **Zone Analysis** — bar chart or map of `Ride_Count` by `Region` from
   `Demand_By_Region`, with a Region filter that also filters the trend line.
4. **Model Comparison** — bar chart of `MAE`/`RMSE` per `Model` from
   `Model_Comparison` — include the naive baseline bar so viewers see the
   improvement in context, not an absolute number in isolation.
5. **Predictions** — line chart combining the tail of `Daily_Demand` with
   `Future_Predictions`, so the forecast visually continues the historical line.
6. **Business Insights** — text boxes summarizing: the weekend effect
   (+38% rides), the trend reversal after April 2023, and the
   recommendation to deploy LSTM.

## 5. Publish

Server menu → Save to Tableau Public → sign in / create a free account.
You'll get a shareable link for the README and your presentation.
