#!/usr/bin/env lua
--  skill_03.lua
--
--  Implements two Apache diagnostics:
--
--  APACHE-05 – Compare KeepAlive enabled vs disabled.
--  APACHE-06 – Compare accessLog enabled vs disabled.
--
--  Usage:
--      lua skill_03.lua <csv_path>
--
--  Output:
--      Lines prefixed by "== APACHE-05 ==" and
--      "== APACHE-06 ==" followed by numeric results
--      and a simple significance assessment.

local ezr = require"ezr"

-- ---- Utilities ------------------------------------------------
-- Load a CSV file and return the table of rows.
local function load_csv(path)
  local src = ezr.csv(path)
  return ezr.Tbl(src)
end

-- Train a regression tree on the full data.
local function train_tree(tbl)
  return ezr.Tree(tbl, tbl.rows)
end

-- Compute predictions for all rows using the tree.
local function compute_predictions(tree, tbl)
  local preds = {}
  for i, row in ipairs(tbl.rows) do
    preds[i] = tree:leaf(tbl, row)
  end
  return preds
end

-- Compute root‑mean‑square residual (sigma).
local function compute_sigma(tree, tbl, preds)
  local sumsq = 0
  for i, row in ipairs(tbl.rows) do
    local actual = row[tbl.cols.y[1].at]
    local r = actual - preds[i]
    sumsq = sumsq + r * r
  end
  return math.sqrt(sumsq / #tbl.rows)
end

-- Gather group statistics for a binary feature.
local function group_stats(tbl, preds, feat_at)
  local groups = { [0]={n=0, sum=0, sumsq=0, actsum=0},
                  [1]={n=0, sum=0, sumsq=0, actsum=0} }

  for i, row in ipairs(tbl.rows) do
    local key = row[feat_at] or 0
    if key ~= nil and (key == 0 or key == 1) then
      local g = groups[key]
      g.n = g.n + 1
      g.sum = g.sum + preds[i]
      g.sumsq = g.sumsq + preds[i] * preds[i]
      g.actsum = g.actsum + row[tbl.cols.y[1].at]
    end
  end
  return groups
end

-- Compute mean and standard deviation for a group.
local function mean_std(group)
  if group.n == 0 then return nil, nil end
  local mean = group.sum / group.n
  local var = (group.sumsq / group.n) - mean * mean
  if var < 0 then var = 0 end
  return mean, math.sqrt(var)
end

-- Build a formatted report for a feature comparison.
local function report(feature_name, groups, sigma, feature_at, tbl)
  local g1 = groups[1]
  local g0 = groups[0]
  if not g1 or not g0 or g1.n == 0 or g0.n == 0 then
    return { string.format("Insufficient data for feature '%s'", feature_name) }
  end

  local mean1, std1 = mean_std(g1)
  local mean0, std0 = mean_std(g0)
  local actmean1 = g1.actsum / g1.n
  local actmean0 = g0.actsum / g0.n

  local diff = mean1 - mean0
  local se = math.sqrt(sigma * sigma / g1.n + sigma * sigma / g0.n)
  local t = diff / se
  local significant = math.abs(t) > 1.96  -- 95% confidence

  return {
    string.format("Feature '%s':", feature_name),
    string.format("  %s=1  : n=%d  mean pred=%.2f  mean actual=%.2f",
                  feature_name, g1.n, mean1, actmean1),
    string.format("  %s=0  : n=%d  mean pred=%.2f  mean actual=%.2f",
                  feature_name, g0.n, mean0, actmean0),
    string.format("  Difference (1-0) = %.2f", diff),
    string.format("  t‑stat = %.2f", t),
    string.format("  Significant at 0.05? %s",
                  significant and "Yes" or "Inconclusive")
  }
end

-- ---- Main ----------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_03.lua <csv_path>\n")
  os.exit(1)
end

local tbl = load_csv(csv_path)
local tree = train_tree(tbl)

-- Predicted values for every row.
local preds = compute_predictions(tree, tbl)

-- Residual sigma from the full model.
local sigma = compute_sigma(tree, tbl, preds)

-- Find column indices for KeepAlive and accessLog.
local keepalive_at = nil
local accesslog_at = nil
for _, c in ipairs(tbl.cols.all) do
  if c.name == "keepAlive" then keepalive_at = c.at end
  if c.name == "accessLog" then accesslog_at = c.at end
end

-- If either feature is missing, exit gracefully.
if not keepalive_at or not accesslog_at then
  io.stderr:write("Required columns missing.\n")
  os.exit(1)
end

-- Group statistics for each feature.
local keep_groups = group_stats(tbl, preds, keepalive_at)
local access_groups = group_stats(tbl, preds, accesslog_at)

-- Report APACHE-05
print("== APACHE-05 ==")
for _, line in ipairs(report("keepAlive", keep_groups, sigma, keepalive_at, tbl)) do
  print(line)
end
print()

-- Report APACHE-06
print("== APACHE-06 ==")
for _, line in ipairs(report("accessLog", access_groups, sigma, accesslog_at, tbl)) do
  print(line)
end
