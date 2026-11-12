#!/usr/bin/env lua
-- Skill 09: Detect anomalous vehicle records
-- CAR-17: Flag vehicles with unusually distinct combinations
-- CAR-18: Flag vehicles whose MPG or weight is far from expected
-- Usage: lua skill_09.lua <csv path>

local ezr = require("ezr")
local show, round = ezr.show, ezr.round

-- -----------------------------------------------------------------
-- Common helpers (copied from earlier skills)
-- -----------------------------------------------------------------
local function col_by_name(tbl, name)
  for _,c in ipairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column " .. name .. " not found")
end

local function minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _,r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  return minv, maxv
end

-- -----------------------------------------------------------------
-- Build a simple regression tree for a numeric target
-- -----------------------------------------------------------------
local function build_tree(tbl, target_col, max_depth)
  local Y = function(row) return row[target_col.at] end
  return ezr.Tree(tbl, tbl.rows, Y, nil, max_depth or 4)
end

local function leaf_node(tree, tbl, row)
  local node = tree
  while node.at do
    local col = tbl.cols.all[node.at]
    if col:holds(row[col.at], node.v) then
      node = node.yes
    else
      node = node.no
    end
  end
  return node
end

-- -----------------------------------------------------------------
-- CAR-17: Anomalous combinations
-- -----------------------------------------------------------------
print("== CAR-17 ==")
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_09.lua <csv path>\n")
  os.exit(1)
end
local tbl = ezr.Tbl(ezr.csv(path))
local rows = tbl.rows

-- Compute nearest neighbour distance for each row
local nn_dists = {}
for i,r1 in ipairs(rows) do
  local best = math.huge
  for j,r2 in ipairs(rows) do
    if i ~= j then
      local d = tbl:distx(r1, r2)
      if d < best then best = d end
    end
  end
  nn_dists[i] = best
end

-- Statistics on nearest distances
local sum, sumsq = 0, 0
for _,d in ipairs(nn_dists) do
  sum = sum + d
  sumsq = sumsq + d*d
end
local n = #nn_dists
local mean = sum / n
local var = (sumsq - n*mean*mean) / (n-1)
local sd = math.sqrt(var)
local thresh = mean + 2*sd

-- Report rows beyond threshold
local flagged = {}
for i,d in ipairs(nn_dists) do
  if d > thresh then
    table.insert(flagged, {idx=i, dist=d})
  end
end

print(string.format("Total rows: %d, mean NN distance: %.3f, sd: %.3f", n, mean, sd))
print(string.format("Threshold for anomaly: %.3f (mean+2sd)", thresh))
print("\nAnomalous records (nearest neighbour distance > threshold):")
for _,f in ipairs(flagged) do
  local r = rows[f.idx]
  print(string.format("#%d  Model=%s  NN_dist=%.3f",
        f.idx, tostring(r[3] or "?"), f.dist))
end
if #flagged == 0 then print("  None detected.") end

-- -----------------------------------------------------------------
-- CAR-18: Unexpected MPG or weight
-- -----------------------------------------------------------------
print("\n== CAR-18 ==")
local mpg_col = col_by_name(tbl, "Mpg+")
local w_col   = col_by_name(tbl, "Lbs-")

-- Build regression trees
local t_mpg = build_tree(tbl, mpg_col)
local t_wt  = build_tree(tbl, w_col)

local flagged_mpg = {}
local flagged_wt = {}

for _,r in ipairs(rows) do
  -- MPG check
  if r[mpg_col.at] ~= "?" then
    local node = leaf_node(t_mpg, tbl, r)
    local est = node.mu
    local diff = math.abs(est - r[mpg_col.at])
    if diff > 2*node.loss then
      table.insert(flagged_mpg, {row=r, est=est, diff=diff})
    end
  end
  -- Weight check
  if r[w_col.at] ~= "?" then
    local node = leaf_node(t_wt, tbl, r)
    local est = node.mu
    local diff = math.abs(est - r[w_col.at])
    if diff > 2*node.loss then
      table.insert(flagged_wt, {row=r, est=est, diff=diff})
    end
  end
end

print("\nRows with abnormal MPG (actual vs predicted):")
print(string.format("%-3s %-10s %-6s %-6s", "No", "Model", "Actual", "Diff"))
for i,f in ipairs(flagged_mpg) do
  print(string.format("%-3d %-10s %-6.1f %-6.1f",
        i, tostring(f.row[3] or "?"), f.row[mpg_col.at], f.diff))
end
if #flagged_mpg == 0 then print("  None detected.") end

print("\nRows with abnormal Weight (actual vs predicted):")
print(string.format("%-3s %-10s %-6s %-6s", "No", "Model", "Actual", "Diff"))
for i,f in ipairs(flagged_wt) do
  print(string.format("%-3d %-10s %-6.0f %-6.0f",
        i, tostring(f.row[3] or "?"), f.row[w_col.at], f.diff))
end
if #flagged_wt == 0 then print("  None detected.") end
