-- skill_01.lua
-- Summary and outlier analysis for Apache server configurations
-- Usage: lua skill_01.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv = ezr.csv

-- ------------------------------------------------------------------
-- Load the CSV into a table structure
-- ------------------------------------------------------------------
local function load_table(path)
  -- csv() returns an iterator, Tbl will fold it into a TBL
  return ezr.Tbl(csv(path))
end

-- ------------------------------------------------------------------
-- Compute basic statistics for a column (numeric assumed)
-- ------------------------------------------------------------------
local function stats(values)
  local n = #values
  if n == 0 then return nil end
  local sum, sumsq = 0, 0
  local minv, maxv = math.huge, -math.huge
  for _, v in ipairs(values) do
    sum = sum + v
    sumsq = sumsq + v * v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local variance = (sumsq / n) - (mean * mean)
  local sd = math.sqrt(math.max(variance, 0))
  -- median
  local sorted = ezr.sorted(values)
  local median = sorted[math.floor(n / 2) + 1]
  return {min = minv, max = maxv, mean = mean, sd = sd, median = median}
end

-- ------------------------------------------------------------------
-- Determine distinguishing settings relative to the mode of each column
-- ------------------------------------------------------------------
local function distinguishing_settings(tbl, row)
  local diffs = {}
  for at = 1, #tbl.cols.all do
    local col = tbl.cols.all[at]
    local mode = col:mid()  -- works for SYM columns
    if row[at] ~= mode then
      table.insert(diffs, col.name .. "=" .. tostring(row[at]))
    end
  end
  return diffs
end

-- ------------------------------------------------------------------
-- Main entry point
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end

  local tbl = load_table(path)

  -- Performance column: 10th column (Performance-)
  local perf_idx = 10
  local perf_values = {}
  for _, row in ipairs(tbl.rows) do
    local v = row[perf_idx]
    if v ~= "?" then
      table.insert(perf_values, v)
    end
  end

  local perf_stats = stats(perf_values)

  -- APACHE-01: summary
  print("== APACHE-01 ==")
  print(string.format("Typical (median) performance: %d", perf_stats.median))
  print(string.format("Best performance: %d", perf_stats.min))
  print(string.format("Worst performance: %d", perf_stats.max))

  -- APACHE-02: unusually favorable configurations
  print("\n== APACHE-02 ==")
  -- threshold: mean - 1.5 * sd
  local threshold = perf_stats.mean - 1.5 * perf_stats.sd
  local fav_rows = {}
  for _, row in ipairs(tbl.rows) do
    local v = row[perf_idx]
    if v ~= "?" and v <= threshold then
      table.insert(fav_rows, row)
    end
  end

  if #fav_rows == 0 then
    print("No unusually favorable configurations found.")
  else
    for i, row in ipairs(fav_rows) do
      -- print the configuration
      local parts = {}
      for at = 1, #row do
        table.insert(parts, tostring(row[at]))
      end
      print(string.format("Row %d: %s", i, table.concat(parts, ", ")))
      -- distinguishing settings
      local diffs = distinguishing_settings(tbl, row)
      if #diffs > 0 then
        print("  Distinguishing settings: " .. table.concat(diffs, ", "))
      else
        print("  No distinguishing settings.")
      end
    end
  end
end

-- Run
main()
