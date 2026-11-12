#!/usr/bin/env lua
--  skill_02.lua – estimate missing MPG (CAR‑03) and weight (CAR‑04)
--  Usage: lua skill_02.lua <csv_path>
--
--  Depends on the ezr library (ezr.lua & ezr-lib.lua) in the
--  same directory or in the Lua package path.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function find_col(tbl, name)
  for _, c in ipairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column " .. name .. " not found")
end

local function mean_sd(vals)
  local n = #vals
  if n == 0 then return 0, 0 end
  local sum = 0
  for _, v in ipairs(vals) do sum = sum + v end
  local mu = sum / n
  local m2 = 0
  for _, v in ipairs(vals) do
    local d = v - mu
    m2 = m2 + d * d
  end
  local sd = n > 1 and math.sqrt(m2 / (n - 1)) or 0
  return mu, sd
end

local function leaf_node(tree, tbl, row)
  while tree.at do
    local c = tbl.cols.all[tree.at]
    if c:holds(row[c.at], tree.v) then
      tree = tree.yes
    else
      tree = tree.no
    end
  end
  return tree
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_02.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Identify relevant columns
-- ------------------------------------------------------------------
local Lbs_col  = find_col(tbl, "Lbs-")
local Mpg_col  = find_col(tbl, "Mpg+")
local Acc_col  = find_col(tbl, "Acc+")
local Vol_col  = find_col(tbl, "Volume")
local Hp_col   = find_col(tbl, "HpX")
local Model_col= find_col(tbl, "Model")
local Origin_col= find_col(tbl, "origin")

-- ------------------------------------------------------------------
-- CAR-03: Estimate missing MPG
-- ------------------------------------------------------------------
print("== CAR-03 ==")

local k = 3  -- number of nearest neighbours
-- Pre‑compute rows with known MPG
local known_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[Mpg_col.at] ~= "?" then
    table.insert(known_rows, r)
  end
end

local mpg_estimates = {}
for _, r in ipairs(tbl.rows) do
  if r[Mpg_col.at] == "?" then
    -- compute distances to all known rows
    local dlist = {}
    for _, nr in ipairs(known_rows) do
      local d = tbl:distx(r, nr)
      dlist[#dlist+1] = {d = d, mpg = nr[Mpg_col.at]}
    end
    table.sort(dlist, function(a,b) return a.d < b.d end)
    local neigh = {}
    for i=1,math.min(k,#dlist) do
      neigh[#neigh+1] = dlist[i].mpg
    end
    local mu, sd = mean_sd(neigh)
    mpg_estimates[#mpg_estimates+1] = {
      row = r,
      mpg = mu,
      err = sd
    }
  end
end

if #mpg_estimates == 0 then
  print("No missing MPG values found.")
else
  for i=1,math.min(5,#mpg_estimates) do
    local e = mpg_estimates[i]
    print(string.format(
      "Clndrs=%s Volume=%s HpX=%s Model=%s Origin=%s Lbs-=%s Acc+=%s",
      ezr.show(e.row[1]), ezr.show(e.row[2]), ezr.show(e.row[3]),
      ezr.show(e.row[4]), ezr.show(e.row[5]), ezr.show(e.row[6]),
      ezr.show(e.row[7])))
    print(string.format(
      "  Estimated MPG: %.2f ± %.2f",
      ezr.round(e.mpg), ezr.round(e.err)))
  end
end

-- ------------------------------------------------------------------
-- CAR-04: Estimate weight
-- ------------------------------------------------------------------
print("\n== CAR-04 ==")

-- Build a regression tree for weight
local tree_weight = ezr.Tree(tbl, tbl.rows, function(r)
  return r[Lbs_col.at]
end)

local weight_estimates = {}
for _, r in ipairs(tbl.rows) do
  local pred = tree_weight:leaf(tbl, r)
  local node = leaf_node(tree_weight, tbl, r)
  -- gather actual weights in the leaf
  local leaf_weights = {}
  for _, lr in ipairs(node.here.rows) do
    table.insert(leaf_weights, lr[Lbs_col.at])
  end
  local mu, sd = mean_sd(leaf_weights)
  weight_estimates[#weight_estimates+1] = {
    row = r,
    weight = mu,
    err = sd
  }
end

for i=1,math.min(5,#weight_estimates) do
  local e = weight_estimates[i]
  print(string.format(
    "Clndrs=%s Volume=%s HpX=%s Model=%s Origin=%s Lbs-=%s Acc+=%s Mpg+=%s",
    ezr.show(e.row[1]), ezr.show(e.row[2]), ezr.show(e.row[3]),
    ezr.show(e.row[4]), ezr.show(e.row[5]), ezr.show(e.row[6]),
    ezr.show(e.row[7]), ezr.show(e.row[8])))
  print(string.format(
    "  Estimated Weight: %.0f ± %.0f",
    ezr.round(e.weight), ezr.round(e.err)))
end
