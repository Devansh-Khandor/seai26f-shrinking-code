#!/usr/bin/env lua
-- Skill 09 – Outlier flagging & consistency checks
-- Usage:  lua skill_09.lua <csv_path>

local ezr   = require("ezr")
local round = ezr.round
local sorted = ezr.sorted
local keysort = ezr.keysort

-- ------------------------------------------------------------------
-- Helper: find a column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- ------------------------------------------------------------------
-- Normalisation helper (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- Read CSV
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: missing CSV path\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Columns we need for distance calculations (x)
local weight_col = get_col(tbl, "Lbs-")
local mpg_col    = get_col(tbl, "Mpg+")
local acc_col    = get_col(tbl, "Acc+")
local model_col  = get_col(tbl, "Model")
local cyl_col    = get_col(tbl, "Clndrs")   -- for a bit of extra context

-- ------------------------------------------------------------------
-- Helper: compute minimal distance of a row to all others
local function min_dist(row)
  local min = nil
  for _, r in ipairs(tbl.rows) do
    if r ~= row then
      local d = tbl:distx(row, r)
      if min == nil or d < min then min = d end
    end
  end
  return min or 0
end

-- ------------------------------------------------------------------
-- CAR-17: Flag unusually distant vehicles
print("== CAR-17 ==")

-- Compute min distances for all rows
local dist_list = {}
for _, r in ipairs(tbl.rows) do
  table.insert(dist_list, {row=r, d=min_dist(r)})
end

-- Statistics on distances
table.sort(dist_list, function(a,b) return a.d < b.d end)
local d_vals = sorted(map(dist_list, "d"))
local d_mean = 0
for _,v in ipairs(d_vals) do d_mean = d_mean + v end
d_mean = d_mean / #d_vals
local d_std = 0
for _,v in ipairs(d_vals) do d_std = d_std + (v - d_mean)^2 end
d_std = math.sqrt(d_std / #d_vals)

-- Threshold: mean + 2*std
local threshold = d_mean + 2 * d_std

-- Collect outliers
local outliers = {}
for _, item in ipairs(dist_list) do
  if item.d > threshold then
    table.insert(outliers, item)
  end
end

if #outliers == 0 then
  print("No records flagged as unusually different.")
else
  -- Show top 3 farthest
  for i=1,math.min(3, #outliers) do
    local r = outliers[i].row
    print(string.format(
      "%d) Model=%s | Cyl=%s | Dist=%.3f",
      i, r[model_col.at], r[cyl_col.at], round(outliers[i].d,3)))
  end
end

-- ------------------------------------------------------------------
-- Helper: predict expected value from k nearest neighbours
local function expected_from_neighbors(row, target_col, k)
  local dlist = {}
  for _, r in ipairs(tbl.rows) do
    if r ~= row then
      local d = tbl:distx(row, r)
      table.insert(dlist, {d=d, r=r})
    end
  end
  table.sort(dlist, function(a,b) return a.d < b.d end)
  local sum, cnt = 0, 0
  for i=1,math.min(k, #dlist) do
    local v = dlist[i].r[target_col.at]
    if v ~= "?" then
      sum = sum + v
      cnt = cnt + 1
    end
  end
  if cnt == 0 then return nil end
  return sum / cnt
end

-- ------------------------------------------------------------------
-- CAR-18: Flag records with large discrepancy
print("\n== CAR-18 ==")

local k_neigh = 5
local flag_rows = {}

for _, r in ipairs(tbl.rows) do
  -- MPG consistency
  local exp_mpg = expected_from_neighbors(r, mpg_col, k_neigh)
  local diff_mpg = nil
  if exp_mpg and r[mpg_col.at] ~= "?" then
    diff_mpg = math.abs(r[mpg_col.at] - exp_mpg)
    if diff_mpg > 5 then  -- arbitrary threshold: 5 mpg units
      table.insert(flag_rows, {
        row=r,
        type="MPG",
        observed=r[mpg_col.at],
        expected=round(exp_mpg,1),
        diff=round(diff_mpg,1)
      })
    end
  end

  -- Weight consistency
  local exp_w = expected_from_neighbors(r, weight_col, k_neigh)
  local diff_w = nil
  if exp_w and r[weight_col.at] ~= "?" then
    diff_w = math.abs(r[weight_col.at] - exp_w)
    if diff_w > 200 then  -- arbitrary threshold: 200 kg
      table.insert(flag_rows, {
        row=r,
        type="Weight",
        observed=r[weight_col.at],
        expected=round(exp_w,1),
        diff=round(diff_w,1)
      })
    end
  end
end

if #flag_rows == 0 then
  print("No records flagged for inconsistency.")
else
  for i, f in ipairs(flag_rows) do
    print(string.format(
      "%d) Model=%s | %s mismatch | Observed=%s | Expected=%s | Diff=%s",
      i, f.row[model_col.at], f.type,
      f.observed, f.expected, f.diff))
  end
end
