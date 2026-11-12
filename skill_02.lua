#!/usr/bin/env lua
-- skill_02.lua
-- Implements the requested APACHE-03 and APACHE-04 analyses.
--
-- Usage: lua skill_02.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- Helper: display numbers using ezr's rounding
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Build the table from the CSV file.
local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_02.lua <csv path>")

local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]  -- the single goal column

-- Pre‑collect rows with their indices for easier access
local rows = tbl.rows
local nrows = #rows

-- ---------- APACHE‑03 ----------

-- Find the k nearest neighbours (by setting distance) for a row.
local function k_nearest(row, k)
  local d = {}
  for j, r in ipairs(rows) do
    if r ~= row then
      push(d, {dist=tbl:distx(row, r), row=r})
    end
  end
  local sorted = ezr.keysort(d, "dist")
  local res = {}
  for i = 1, math.min(k, #sorted) do
    push(res, sorted[i].row)
  end
  return res
end

-- Estimate performance from neighbours (inverse distance weighting).
local function estimate(row, k)
  local neigh = k_nearest(row, k)
  local wsum = 0
  local num = 0
  for _, r in ipairs(neigh) do
    local d = tbl:distx(row, r) + 1e-6
    local w = 1 / d
    wsum = wsum + w * r[perf_col.at]
    num = num + w
  end
  return wsum / num
end

-- Compute predictions and errors for all rows
local predictions = {}
local errors = {}
for i, r in ipairs(rows) do
  local pred = estimate(r, 5)
  push(predictions, {row=i, pred=pred, actual=r[perf_col.at]})
  push(errors, math.abs(pred - r[perf_col.at]))
end

local sorted_err = ezr.sorted(errors)
local median_abs_error = sorted_err[math.floor(#errors/2)+1]

-- Find worst error
local worst = ezr.sorted(ezr.map(predictions,
  function(p) return {err=math.abs(p.pred-p.actual), p=p} end),
  "err")[1].p

print("== APACHE-03 ==")
print(string.format("Median absolute prediction error: %s", display(median_abs_error)))
print(string.format("Example prediction (row 1): actual=%s, predicted=%s",
      display(rows[1][perf_col.at]), display(predictions[1].pred)))
print(string.format("Worst prediction (row %d): actual=%s, predicted=%s, error=%s",
      worst.row, display(worst.actual), display(worst.pred), display(math.abs(worst.pred-worst.actual))))

-- ---------- APACHE‑04 ----------

-- Identify all pairs that differ in exactly one setting.
local function diff_one_setting(r1, r2)
  local diff = 0
  for _,c in ipairs(tbl.cols.all) do
    if c ~= perf_col then
      local v1, v2 = r1[c.at], r2[c.at]
      if v1 ~= v2 and v1 ~= "?" and v2 ~= "?" then
        diff = diff + 1
        if diff > 1 then return false end
      end
    end
  end
  return diff == 1
end

local deltas = {}
for i = 1, nrows-1 do
  for j = i+1, nrows do
    if diff_one_setting(rows[i], rows[j]) then
      local d = rows[j][perf_col.at] - rows[i][perf_col.at]
      push(deltas, {delta=d, i=i, j=j})
    end
  end
end

local ndel = #deltas
if ndel == 0 then
  print("\n== APACHE-04 ==")
  print("No configuration pairs differ by exactly one setting.")
else
  local sum, sumsq = 0, 0
  for _,d in ipairs(deltas) do
    sum = sum + d.delta
    sumsq = sumsq + d.delta * d.delta
  end
  local mean = sum / ndel
  local variance = (sumsq - ndel * mean * mean) / (ndel - 1)
  local stddev = math.sqrt(variance)

  local sorted_by_delta = ezr.sorted(deltas, "delta")
  local maxpos = sorted_by_delta[#sorted_by_delta]
  local minpos = sorted_by_delta[1]

  print("\n== APACHE-04 ==")
  print(string.format("Mean performance difference between one‑setting variants: %s",
        display(mean)))
  print(string.format("Standard deviation of differences: %s", display(stddev)))
  print(string.format("Largest positive difference: delta=%s between rows %d and %d",
        display(maxpos.delta), maxpos.i, maxpos.j))
  print(string.format("Largest negative difference: delta=%s between rows %d and %d",
        display(minpos.delta), minpos.i, minpos.j))
end
