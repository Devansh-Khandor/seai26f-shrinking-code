#!/usr/bin/env lua
-- Skill 04 – Apache performance monitoring and incremental update
--
-- Usage:  lua skill_04.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.

local ezr = require("ezr")

-- Helper: convert a row to CSV string (excluding goal column)
local function row_to_csv(row, goal_at)
  local parts = {}
  for i, v in ipairs(row) do
    if i ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Main --------------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_04.lua <csv path>")
    os.exit(1)
  end

  -- Load data into a table (header is processed by ezr)
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Identify performance column
  local perf_col = tbl.cols.y[1]
  local perf_at  = perf_col.at

  print("== APACHE-07 ==")

  -- Incremental alerting for unusually poor performance
  local n, sum, sumsq = 0, 0, 0
  for idx, r in ipairs(tbl.rows) do
    local perf = r[perf_at]

    if n > 1 then
      local mean_prev = sum / n
      local var_prev = (sumsq - sum * sum / n) / (n - 1)
      local std_prev = math.sqrt(math.max(var_prev, 0))

      if std_prev > 0 then
        local z = (perf - mean_prev) / std_prev
        if z > 2 then
          print(string.format(
            "Alert: Row %d performance %.1f is %.1fσ above mean (%.1f)",
            idx, perf, z, mean_prev))
        end
      end
    end

    -- Update running statistics
    n, sum, sumsq = n + 1, sum + perf, sumsq + perf * perf
  end

  -- If no alerts, still provide minimal output
  if n <= 1 then
    print("No previous data to compare against.")
  end

  print("\n== APACHE-08 ==")

  -- Performance summary (mean and std) for the entire data set
  local perf_mean = perf_col:mid()
  local perf_std  = perf_col:div()
  print(string.format("Overall performance: mean = %.1f, std = %.1f", perf_mean, perf_std))

  -- Best configurations shortlist (top 25% lowest performance)
  local sorted_rows = ezr.keysort(tbl.rows, function(r) return r[perf_at] end)
  local top_n = math.max(1, math.floor(#sorted_rows * 0.25))
  print(string.format("Top %d configurations (lowest performance):", top_n))
  for i = 1, top_n do
    local r = sorted_rows[i]
    print(string.format("[%d] %s", i, row_to_csv(r, perf_at)))
  end

  -- Updated summary statistics for each column
  print("\nUpdated column summaries:")
  for _, c in ipairs(tbl.cols.all) do
    if c.at ~= perf_at then
      local mid = c:mid()
      local div = c:div()
      print(string.format("- %s: mean=%.2f, std=%.2f", c.name, mid, div))
    end
  end
end

main(arg[1])
