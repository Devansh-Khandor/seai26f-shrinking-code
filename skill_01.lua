#!/usr/bin/env lua
-- Skill 01 – Apache performance summary & best configurations
--
-- Usage:  lua skill_01.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.

local ezr = require("ezr")

-- helper: format a row as a csv line, excluding the goal column
local function row_to_csv(row, goal_at)
  local parts = {}
  for k, v in ipairs(row) do
    if k ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- main --------------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_01.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- The goal column (performance)
  local perf_col = tbl.cols.y[1]          -- only one goal in this data
  local perf_at  = perf_col.at
  local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_at] end)
  local sorted_perf = ezr.sorted(perf_vals)

  -- === APACHE-01: summary ==============================================
  local n = #sorted_perf
  local median
  if n % 2 == 1 then
    median = sorted_perf[(n + 1) / 2]
  else
    median = (sorted_perf[n / 2] + sorted_perf[n / 2 + 1]) / 2
  end
  local best = sorted_perf[n]
  local worst = sorted_perf[1]

  print("== APACHE-01 ==")
  print(string.format("Typical (median) performance: %.0f", median))
  print(string.format("Best performance: %.0f", best))
  print(string.format("Worst performance: %.0f", worst))
  print()  -- blank line for readability

  -- === APACHE-02: best configurations & distinguishing settings =====
  -- Choose the top quarter (or at least one) best rows
  local top_n = math.max(1, math.floor(n * 0.25))
  local best_rows = {}
  for i = 1, top_n do
    table.insert(best_rows, tbl.rows[ezr.keysort(tbl.rows, function(r) return r[perf_at] end)[i]])
  end

  print("== APACHE-02 ==")
  print(string.format("Top %d configurations (lowest performance):", top_n))
  for i, r in ipairs(best_rows) do
    print(string.format("[%d] %s", i, row_to_csv(r, perf_at)))
  end

  -- Compute mean of each column overall and in best group
  local overall_means = {}
  local overall_sds = {}
  for _, c in ipairs(tbl.cols.all) do
    overall_means[c.at] = c:mid()
    overall_sds[c.at]   = c:div()
  end

  -- Sum in best group
  local best_sums = {}
  local best_counts = {}
  for _, c in ipairs(tbl.cols.all) do
    best_sums[c.at] = 0
    best_counts[c.at] = 0
  end
  for _, r in ipairs(best_rows) do
    for _, c in ipairs(tbl.cols.all) do
      best_sums[c.at] = best_sums[c.at] + r[c.at]
      best_counts[c.at] = best_counts[c.at] + 1
    end
  end
  local best_means = {}
  for _, c in ipairs(tbl.cols.all) do
    best_means[c.at] = best_sums[c.at] / best_counts[c.at]
  end

  -- Find distinguishing settings (lower mean than overall)
  print("\nDistinguishing settings (mean lower in best configs):")
  for _, c in ipairs(tbl.cols.all) do
    if c.at ~= perf_at then
      local diff = overall_means[c.at] - best_means[c.at]
      if diff > 0 then
        print(string.format("- %s: best mean = %.2f, overall mean = %.2f (Δ=%.2f)",
                            c.name, best_means[c.at], overall_means[c.at], diff))
      end
    end
  end
end

main(arg[1])
