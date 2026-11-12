#!/usr/bin/env lua
-- Skill 03 – Apache performance comparison of key settings
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

------------------------------------------------------------------
-- Get the CSV file path from the command line
------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_03.lua <csv path>")

------------------------------------------------------------------
-- Load the data into an ezr table
------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

------------------------------------------------------------------
-- Helper functions -------------------------------------------------
------------------------------------------------------------------
local function find_col_at(name)
  for at, col in ipairs(tbl.cols.all) do
    if col.name == name then return at end
  end
  error("Column '" .. name .. "' not found")
end

local function mean(lst)
  local sum = 0
  for _, v in ipairs(lst) do sum = sum + v end
  return sum / (#lst > 0 and #lst or 1)
end

local function stddev(lst, m)
  local sum = 0
  for _, v in ipairs(lst) do sum = sum + (v - m)^2 end
  return (#lst > 1 and math.sqrt(sum / (#lst - 1)) or 0)
end

------------------------------------------------------------------
-- Locate column indices
------------------------------------------------------------------
local perf_at      = tbl.cols.klass.at      -- performance column (goal)
local keep_at      = find_col_at"keepAlive"
local access_at    = find_col_at"accessLog"

------------------------------------------------------------------
-- Split data into groups
------------------------------------------------------------------
local keep_vals   = {}
local no_keep_vals = {}

local access_vals   = {}
local no_access_vals = {}

for _, r in ipairs(tbl.rows) do
  local perf = r[perf_at]

  -- KeepAlive groups
  if r[keep_at] == 1 then
    table.insert(keep_vals, perf)
  elseif r[keep_at] == 0 then
    table.insert(no_keep_vals, perf)
  end

  -- AccessLog groups
  if r[access_at] == 1 then
    table.insert(access_vals, perf)
  elseif r[access_at] == 0 then
    table.insert(no_access_vals, perf)
  end
end

------------------------------------------------------------------
-- APACHE-05 : KeepAlive comparison
------------------------------------------------------------------
print("== APACHE-05 ==")

local keep_mean   = mean(keep_vals)
local keep_sd     = stddev(keep_vals, keep_mean)

local no_mean     = mean(no_keep_vals)
local no_sd       = stddev(no_keep_vals, no_mean)

local d_keep      = ezr.cohen(keep_vals, no_keep_vals)
local same_keep   = not ezr.same(keep_vals, no_keep_vals)

print(string.format("KeepAlive enabled  mean: %s (SD %s)",
                    ezr.round(keep_mean), ezr.round(keep_sd)))
print(string.format("KeepAlive disabled mean: %s (SD %s)",
                    ezr.round(no_mean), ezr.round(no_sd)))
print(string.format("Cohen's d: %s", ezr.round(d_keep)))
if same_keep then
  print("Inconclusive evidence: no meaningful difference observed.")
else
  print("Significant difference observed between enabled/disabled KeepAlive.")
end

------------------------------------------------------------------
-- APACHE-06 : AccessLog comparison
------------------------------------------------------------------
print("\n== APACHE-06 ==")

local access_mean   = mean(access_vals)
local access_sd     = stddev(access_vals, access_mean)

local no_access_mean = mean(no_access_vals)
local no_access_sd   = stddev(no_access_vals, no_access_mean)

local d_access     = ezr.cohen(access_vals, no_access_vals)
local same_access  = not ezr.same(access_vals, no_access_vals)

print(string.format("AccessLog enabled  mean: %s (SD %s)",
                    ezr.round(access_mean), ezr.round(access_sd)))
print(string.format("AccessLog disabled mean: %s (SD %s)",
                    ezr.round(no_access_mean), ezr.round(no_access_sd)))
print(string.format("Cohen's d: %s", ezr.round(d_access)))
if same_access then
  print("Inconclusive evidence: no consistent performance advantage.")
else
  print("One group consistently outperforms the other.")
end
