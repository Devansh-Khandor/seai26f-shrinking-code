#!/usr/bin/env lua
-- Skill 02: Performance estimation and difference analysis
local ezr = require "ezr"

-- Utility to pretty‑print numbers/values
local function pretty(v) return ezr.show(v) end

-- Build a decision tree over the whole table
local function train_tree(tbl)
  return ezr.Tree(tbl, tbl.rows)
end

-- Predict the performance of a row using the tree
local function predict(tree, tbl, row)
  return tree:leaf(tbl, row)
end

-- Compute prediction error statistics for all rows
local function prediction_stats(tree, tbl)
  local perf_at = tbl.cols.y[1].at
  local sq, n, sum, sumsq = 0, 0, 0, 0
  for _, r in ipairs(tbl.rows) do
    local actual = r[perf_at]
    local pred   = predict(tree, tbl, r)
    local err    = actual - pred
    n = n + 1
    sum = sum + err
    sumsq = sumsq + err * err
  end
  local mean_err = sum / n
  local rmse = math.sqrt(sumsq / n)
  return {n=n, mean_err=mean_err, rmse=rmse}
end

-- Find all pairs of rows that differ by exactly one setting
-- and compute statistics on their performance differences
local function difference_stats(tbl)
  local perf_at = tbl.cols.y[1].at
  local diffs = {}
  local rows = tbl.rows
  local m = #rows
  for i = 1, m-1 do
    local r1 = rows[i]
    for j = i+1, m do
      local r2 = rows[j]
      local diff_count = 0
      for _, c in ipairs(tbl.cols.x) do
        local v1, v2 = r1[c.at], r2[c.at]
        if v1 ~= v2 and v1 ~= "?" and v2 ~= "?" then
          diff_count = diff_count + 1
        end
      end
      if diff_count == 1 then
        local d = r1[perf_at] - r2[perf_at]
        diffs[#diffs+1] = d
      end
    end
  end

  local n = #diffs
  if n == 0 then
    return {n=0}
  end

  local sum, sumsq = 0, 0
  for _, d in ipairs(diffs) do
    sum = sum + d
    sumsq = sumsq + d*d
  end
  local mean_diff = sum / n
  local sd_diff = math.sqrt((sumsq - sum*sum/n) / n)
  return {n=n, mean=mean_diff, sd=sd_diff}
end

-- Main routine
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_02.lua <csv path>")

  -- Load the data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Train a tree on the whole table
  local tree = train_tree(tbl)

  -- === APACHE-03 ===
  print("== APACHE-03 ==")
  local pstats = prediction_stats(tree, tbl)
  print("Predictions over all rows:")
  print("  Rows:", pretty(pstats.n))
  print("  Mean prediction error:", pretty(pstats.mean_err))
  print("  RMSE:", pretty(pstats.rmse))

  -- === APACHE-04 ===
  print("\n== APACHE-04 ==")
  local dstats = difference_stats(tbl)
  if dstats.n == 0 then
    print("  No pairs differ by exactly one setting.")
  else
    print("Pairs differing by one setting:", pretty(dstats.n))
    print("  Mean performance difference:", pretty(dstats.mean))
    print("  Std dev of difference:", pretty(dstats.sd))
  end
end

-- Execute if this file is run directly
if not debug.getinfo(1,"S").short_src:match("[/\\]skill_02.lua$") then
  main()
end
