#!/usr/bin/env lua
-- skill_01.lua
-- Implements the requested APACHE-01 and APACHE-02 analyses.
--
-- Usage: lua skill_01.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- Helper: median of a numeric list (assumes list is non‑empty).
local function median(list)
  local s = ezr.sorted(list)
  local n = #s
  if n % 2 == 1 then
    return s[(n + 1) // 2]
  else
    return (s[n // 2] + s[n // 2 + 1]) / 2
  end
end

-- Helper: quartiles (returns Q1, Q3)
local function quartiles(list)
  local s = ezr.sorted(list)
  local n = #s
  local q1 = s[(n + 3) // 4]   -- floor((n+3)/4)
  local q3 = s[(3 * n + 1) // 4]  -- floor((3n+1)/4)
  return q1, q3
end

-- Helper: round to the display precision of ezr.
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return v
  end
end

-- Helper: convert a row table to CSV string.
local function row_to_csv(row)
  local parts = {}
  for _,v in ipairs(row) do
    if v == "?" then
      table.insert(parts, "?")
    else
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Load the data file.
local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_01.lua <csv path>")

-- Build the table.
local tbl = ezr.Tbl(ezr.csv(csvfile))

-- Locate the performance column (the only goal column).
local perf_col = tbl.cols.y[1]   -- should be the "Performance-" column.

-- Extract all performance values.
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)

-- Basic statistics.
local perf_mean = perf_col:mid()
local perf_min  = ezr.min(perf_vals)
local perf_max  = ezr.max(perf_vals)
local perf_med  = median(perf_vals)

-- ---- APACHE-01 ----
print("== APACHE-01 ==")
print(string.format("Typical (median): %s", display(perf_med)))
print(string.format("Best (min): %s", display(perf_min)))
print(string.format("Worst (max): %s", display(perf_max)))
print(string.format("Mean: %s", display(perf_mean)))
print()

-- ---- APACHE-02 ----
print("== APACHE-02 ==")

-- Identify unusually favorable configurations.
local q1, q3 = quartiles(perf_vals)
local iqr = q3 - q1
local threshold = perf_med - 1.5 * iqr   -- lower than median by 1.5*IQR
local favorable_rows = {}
for _,row in ipairs(tbl.rows) do
  if row[perf_col.at] <= threshold then
    table.insert(favorable_rows, row)
  end
end

if #favorable_rows == 0 then
  print("No configurations were unusually favorable by the chosen criterion.")
else
  print(string.format("Threshold (median – 1.5*IQR): %s", display(threshold)))
  print(string.format("%d unusually favorable configuration(s) found:", #favorable_rows))
  print()
  for _,row in ipairs(favorable_rows) do
    print(row_to_csv(row))
  end
  print()

  -- Compute distinguishing settings.
  print("Distinguishing settings (overall vs favorable):")
  -- overall values
  local overall_val = {}
  for _,col in ipairs(tbl.cols.all) do
    if col ~= perf_col then
      if getmetatable(col) == ezr.SYM then
        overall_val[col.at] = col:mid()
      else
        overall_val[col.at] = col:mid()
      end
    end
  end
  -- favorable values
  local fav_val = {}
  for _,col in ipairs(tbl.cols.all) do
    if col ~= perf_col then
      if getmetatable(col) == ezr.SYM then
        fav_val[col.at] = ezr.map(favorable_rows, function(r) return r[col.at] end)
        fav_val[col.at] = ezr.SYM:new{has={}}  -- dummy for mode calculation
        -- compute mode manually for favorable subset
        local counts = {}
        for _,v in ipairs(fav_val[col.at]) do
          counts[v] = (counts[v] or 0) + 1
        end
        local best, bestn = nil, -1
        for k,v in pairs(counts) do
          if v > bestn then bestn, best = v, k end
        end
        fav_val[col.at] = best
      else
        fav_val[col.at] = ezr.map(favorable_rows, function(r) return r[col.at] end)
        fav_val[col.at] = ezr.sum(fav_val[col.at], function(v) return v end) / #favorable_rows
      end
    end
  end

  for _,col in ipairs(tbl.cols.all) do
    if col ~= perf_col then
      local overall = overall_val[col.at]
      local fav = fav_val[col.at]
      if type(overall) == "number" and type(fav) == "number" then
        if math.abs(overall - fav) > 1e-6 then
          print(string.format("%-15s: overall= %s  favorable= %s", col.name,
                display(overall), display(fav)))
        end
      else
        if overall ~= fav then
          print(string.format("%-15s: overall= %s  favorable= %s", col.name,
                tostring(overall), tostring(fav)))
        end
      end
    end
  end
end
