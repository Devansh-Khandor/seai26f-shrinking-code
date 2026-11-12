#!/usr/bin/env lua
--  skill_09.lua
--
--  Implements the following Apache diagnostics:
--
--  APACHE-17 – Identify configurations whose recorded performance
--              values are unusually different from what their
--              settings and comparable historical configurations
--              would lead us to expect.
--
--  APACHE-18 – Flag configurations whose combinations of settings
--              differ substantially from the most similar
--              configurations in our historical benchmark records,
--              even when their measured performance is not obviously poor.
--
--  Usage:
--      lua skill_09.lua <csv_path>
--
--  Output:
--      Lines prefixed by "== APACHE-17 ==" and "== APACHE-18 ==",
--      followed by at least three lines of results each.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- 1. Load CSV and build table
-- ------------------------------------------------------------------
local csvpath = arg[1]
if not csvpath then
  io.stderr:write("usage: lua skill_09.lua <csv_path>\n")
  os.exit(1)
end

local src = ezr.csv(csvpath)
local tbl = ezr.Tbl(src)
local perf_col = tbl.cols.y[1]
local n = #tbl.rows

-- ------------------------------------------------------------------
-- 2. Build a regression tree on the full data
-- ------------------------------------------------------------------
local tree = ezr.Tree(tbl, tbl.rows)

-- ------------------------------------------------------------------
-- 3. Compute predictions and residuals
-- ------------------------------------------------------------------
local preds = {}
local residuals = {}
for i, row in ipairs(tbl.rows) do
  local pred = tree:leaf(tbl, row)
  local act = row[perf_col.at]
  preds[i] = pred
  residuals[i] = act - pred
end

-- Compute sigma of residuals
local sumsq = 0
for _, r in ipairs(residuals) do
  sumsq = sumsq + r * r
end
local sigma_res = math.sqrt(sumsq / n)

-- Threshold for unusual performance
local thr_res = 2 * sigma_res

-- Collect candidates
local unusual = {}
for i, r in ipairs(residuals) do
  if math.abs(r) > thr_res then
    table.insert(unusual, {
      idx = i,
      actual = tbl.rows[i][perf_col.at],
      pred = preds[i],
      resid = r,
      z = r / sigma_res
    })
  end
end

-- Sort by absolute residual descending
table.sort(unusual, function(a,b)
  return math.abs(a.resid) > math.abs(b.resid)
end)

-- ------------------------------------------------------------------
-- 4. APACHE-17 : print results
-- ------------------------------------------------------------------
print("== APACHE-17 ==")
if #unusual == 0 then
  print("  No configurations with unusually high residuals.")
else
  local limit = math.min(3, #unusual)
  for i = 1, limit do
    local u = unusual[i]
    print(string.format(
      "  Config %d: actual=%.2f predicted=%.2f residual=%.2f z=%.2f",
      u.idx, u.actual, u.pred, u.resid, u.z))
  end
end

-- ------------------------------------------------------------------
-- 5. APACHE-18 : compute nearest neighbor distances
-- ------------------------------------------------------------------
local nn = {}   -- {idx, dist, neigh}
for i = 1, n do
  local row_i = tbl.rows[i]
  local best = math.huge
  local best_j = nil
  for j = 1, n do
    if j ~= i then
      local d = tbl:distx(row_i, tbl.rows[j])
      if d < best then
        best = d
        best_j = j
      end
    end
  end
  nn[i] = {idx = i, dist = best, neigh = best_j}
end

-- Compute mean & sd of distances
local sumd = 0
for _, v in ipairs(nn) do
  sumd = sumd + v.dist
end
local mean_d = sumd / n
local ssd = 0
for _, v in ipairs(nn) do
  local diff = v.dist - mean_d
  ssd = ssd + diff * diff
end
local sigma_d = math.sqrt(ssd / n)

-- Threshold for unusual distance
local thr_d = mean_d + 2 * sigma_d

-- Collect candidates
local distant = {}
for _, v in ipairs(nn) do
  if v.dist > thr_d then
    table.insert(distant, v)
  end
end

-- Sort by distance descending
table.sort(distant, function(a,b)
  return a.dist > b.dist
end)

-- ------------------------------------------------------------------
-- 6. APACHE-18 : print results
-- ------------------------------------------------------------------
print("\n== APACHE-18 ==")
if #distant == 0 then
  print("  No configurations with unusually large distance to nearest neighbor.")
else
  local limit = math.min(3, #distant)
  for i = 1, limit do
    local d = distant[i]
    print(string.format(
      "  Config %d: nearest distance=%.3f (Config %d)",
      d.idx, d.dist, d.neigh))
  end
end
