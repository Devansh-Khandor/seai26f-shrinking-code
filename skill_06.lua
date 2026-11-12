#!/usr/bin/env lua
-- skill_06.lua
-- CAR-11 & CAR-12: Explain why a car is recommended and compare strengths
-- of two cars with different trade‑offs.
--
-- Usage: lua skill_06.lua <path-to-auto93.csv>
--
-- Requires only the bundled ezr library.
--

local ezr = require("ezr")
local csv  = ezr.csv

-- ------------------------------------------------------------------
-- Helper utilities
-- ------------------------------------------------------------------

-- Compute a numeric list for a column index
local function numeric_values(tbl, col)
  local out = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[col]
    if v ~= "?" and v ~= nil then
      out[#out + 1] = v
    end
  end
  return out
end

-- Mean of a numeric list
local function mean(vals)
  local n = #vals
  if n == 0 then return nil end
  return ezr.sum(vals, function(v) return v end) / n
end

-- Desirability metric: mpg / lbs
local function desirability(row, mpg_idx, lbs_idx)
  local mpg = row[mpg_idx]
  local lbs = row[lbs_idx]
  if mpg == "?" or lbs == "?" or mpg == nil or lbs == nil then
    return nil
  end
  return mpg / lbs
end

-- Find column index by name
local function find_col(tbl, name)
  for _, c in ipairs(tbl.cols.all) do
    if c.name == name then return c.at end
  end
  return nil
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------

local function main(csv_path)
  -- Load the entire table
  local tbl = ezr.Tbl(csv(csv_path))

  -- Find indices of important columns
  local mpg_idx = find_col(tbl, "Mpg+")
  local lbs_idx = find_col(tbl, "Lbs-")
  local model_idx = find_col(tbl, "Model")
  local cyl_idx   = find_col(tbl, "Clndrs")
  local year_idx  = find_col(tbl, "origin")   -- using "origin" as model year placeholder

  if not mpg_idx or not lbs_idx or not model_idx or not cyl_idx then
    error("Required columns missing")
  end

  -- Compute desirability for every row
  local all_d = {}
  for i, r in ipairs(tbl.rows) do
    local d = desirability(r, mpg_idx, lbs_idx)
    if d then
      all_d[#all_d + 1] = {row=i, d=d, r=r}
    end
  end

  -- Sort by desirability descending
  table.sort(all_d, function(a,b) return a.d > b.d end)

  -- Take top 5 vehicles
  local top = {}
  for i = 1, math.min(5, #all_d) do
    top[#top + 1] = all_d[i]
  end

  ------------------------------------------------------------------
  -- CAR-11: Explain recommendations using similar vehicles
  ------------------------------------------------------------------
  print("== CAR-11 ==")
  for i, rec in ipairs(top) do
    local row = rec.r
    local d = rec.d
    local model = row[model_idx]
    -- Gather all rows with same model
    local same = {}
    for _, r in ipairs(tbl.rows) do
      if r[model_idx] == model then
        local dd = desirability(r, mpg_idx, lbs_idx)
        if dd then
          table.insert(same, dd)
        end
      end
    end
    local mean_same = mean(same)
    local pct_diff = ((d - mean_same) / (mean_same + 1e-12)) * 100
    print(string.format(
      "Row %d (%s %s): desirability = %.4f, which is %.1f%% %s than the average of its %s family",
      rec.row, row[year_idx], row[model_idx], d,
      math.abs(pct_diff),
      pct_diff >= 0 and "higher" or "lower",
      #same))
  end

  ------------------------------------------------------------------
  -- CAR-12: Compare two vehicles with different strengths
  ------------------------------------------------------------------
  print("\n== CAR-12 ==")
  if #top < 2 then
    print("Not enough vehicles to compare.")
  else
    local a = top[1].r
    local b = top[2].r
    local da = desirability(a, mpg_idx, lbs_idx)
    local db = desirability(b, mpg_idx, lbs_idx)

    print(string.format("Top car (Row %d): mpg=%.1f, lbs=%.0f, cyl=%d, desirability=%.4f",
      top[1].row, a[mpg_idx], a[lbs_idx], a[cyl_idx], da))
    print(string.format("Second car (Row %d): mpg=%.1f, lbs=%.0f, cyl=%d, desirability=%.4f",
      top[2].row, b[mpg_idx], b[lbs_idx], b[cyl_idx], db))

    -- Trade‑offs
    print("\nTrade‑offs:")
    if a[mpg_idx] > b[mpg_idx] then
      print(string.format("- Car %d has %.1f more MPG (%.2f%% higher).", top[1].row, a[mpg_idx]-b[mpg_idx], ((a[mpg_idx]-b[mpg_idx])/b[mpg_idx])*100))
    else
      print(string.format("- Car %d has %.1f more MPG (%.2f%% higher).", top[2].row, b[mpg_idx]-a[mpg_idx], ((b[mpg_idx]-a[mpg_idx])/a[mpg_idx])*100))
    end

    if a[lbs_idx] < b[lbs_idx] then
      print(string.format("- Car %d is lighter by %.0f lbs (%.2f%% lighter).", top[1].row, b[lbs_idx]-a[lbs_idx], ((b[lbs_idx]-a[lbs_idx])/b[lbs_idx])*100))
    else
      print(string.format("- Car %d is lighter by %.0f lbs (%.2f%% lighter).", top[2].row, a[lbs_idx]-b[lbs_idx], ((a[lbs_idx]-b[lbs_idx])/a[lbs_idx])*100))
    end

    if a[cyl_idx] < b[cyl_idx] then
      print(string.format("- Car %d has %d fewer cylinders than Car %d.", top[1].row, a[cyl_idx]-b[cyl_idx], top[2].row))
    elseif a[cyl_idx] > b[cyl_idx] then
      print(string.format("- Car %d has %d more cylinders than Car %d.", top[1].row, a[cyl_idx]-b[cyl_idx], top[2].row))
    else
      print("- Both cars have the same number of cylinders.")
    end

    print(string.format("\nOverall recommendation: Car %d ranked higher because its %s outweighs its %s when weighted by desirability.",
      top[1].row, (da > db) and "higher MPG and lower weight" or "better balance", (da < db) and "higher weight or lower MPG" or "lower MPG or higher weight"))
  end
end

-- Entry point
if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_06.lua <csv_path>\n")
end
