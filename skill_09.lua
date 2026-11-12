#!/usr/bin/env lua
-- Skill 09: CAR-17 & CAR-18
-- Usage: lua skill_09.lua <csv path>
--
-- CAR-17: flag unusually different cars
-- CAR-18: flag cars with suspect weight or mpg

local ezr = require("ezr")
local csv  = ezr.csv
local sorted = ezr.sorted
local round = ezr.round
local keysort = ezr.keysort

-- ------------------------------------------------------------------
-- Read data
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_09.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

local rows = {}
for r in rows_iter do
  table.insert(rows, r)
end

-- ------------------------------------------------------------------
-- Identify numeric columns (any field whose first value is a number)
local numeric_cols = {}
for _, col in ipairs(headers) do
  local v = rows[1][col]
  if type(v) == "number" then
    table.insert(numeric_cols, col)
  end
end

-- Compute min/max for each numeric column for normalisation
local col_min, col_max = {}, {}
for _, col in ipairs(numeric_cols) do
  local mn, mx = math.huge, -math.huge
  for _, r in ipairs(rows) do
    local v = r[col]
    if v < mn then mn = v end
    if v > mx then mx = v end
  end
  col_min[col], col_max[col] = mn, mx
end

-- Normalisation helper
local function norm(v, mn, mx)
  return mx == mn and 0 or (v - mn) / (mx - mn)
end

-- Build normalized vectors for all rows
local norm_rows = {}
for _, r in ipairs(rows) do
  local nv = {}
  for _, col in ipairs(numeric_cols) do
    nv[col] = norm(r[col], col_min[col], col_max[col])
  end
  table.insert(norm_rows, nv)
end

-- ------------------------------------------------------------------
-- CAR-17 : flag unusual cars
print("== CAR-17 ==")

-- Compute centroid of normalized space
local centroid = {}
for _, col in ipairs(numeric_cols) do
  local sum = 0
  for _, nv in ipairs(norm_rows) do
    sum = sum + nv[col]
  end
  centroid[col] = sum / #norm_rows
end

-- Distance of each row to centroid (euclidean)
local distances = {}
for i, nv in ipairs(norm_rows) do
  local d2 = 0
  for _, col in ipairs(numeric_cols) do
    local diff = nv[col] - centroid[col]
    d2 = d2 + diff * diff
  end
  local dist = math.sqrt(d2)
  distances[i] = dist
end

-- Mean and std of distances
local sumd, sumdsq = 0, 0
for _, d in ipairs(distances) do
  sumd = sumd + d
  sumdsq = sumdsq + d * d
end
local n = #distances
local mean_d = sumd / n
local std_d = math.sqrt((sumdsq - sumd*sumd/n)/(n-1))

-- Threshold for unusualness: mean + 2*std
local thresh_d = mean_d + 2*std_d

-- Collect unusual indices
local unusual = {}
for i, d in ipairs(distances) do
  if d > thresh_d then
    table.insert(unusual, {idx=i, dist=d})
  end
end
-- Sort by distance descending
unusual = sorted(unusual, function(a,b) return a.dist > b.dist end)

-- Print top 5 unusual cars
local top = math.min(5, #unusual)
print(string.format(
  "Idx  Dist  Clndrs  Volume  HpX  Model  origin  Lbs-  Acc+  Mpg+"))
for i=1, top do
  local r = rows[unusual[i].idx]
  print(string.format(
    "%3d  %5.3f  %-7s %-7s %-5s %-6s %-6s %-7s %-5s %-4s",
    unusual[i].idx, unusual[i].dist,
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"]))
end
print()

-- ------------------------------------------------------------------
-- CAR-18 : flag cars with suspect weight or mpg
print("== CAR-18 ==")

-- Function to find k nearest neighbours (by Euclidean distance in norm space)
local function k_neighbors(idx, k)
  local target = norm_rows[idx]
  local neigh = {}
  for j, nv in ipairs(norm_rows) do
    if j ~= idx then
      local d2 = 0
      for _, col in ipairs(numeric_cols) do
        local diff = nv[col] - target[col]
        d2 = d2 + diff * diff
      end
      local d = math.sqrt(d2)
      table.insert(neigh, {idx=j, dist=d})
    end
  end
  return sorted(neigh, function(a,b) return a.dist < b.dist end)
end

-- For each row, predict weight and mpg from neighbours
local k = 5
local resid_weight, resid_mpg = {}, {}
for i=1,#rows do
  local neigh = k_neighbors(i, k)
  local wsum, msum = 0, 0
  for j=1, math.min(k,#neigh) do
    local r = rows[neigh[j].idx]
    wsum = wsum + r["Lbs-"]
    msum = msum + r["Mpg+"]
  end
  local pred_w = wsum / k
  local pred_m = msum / k
  local r = rows[i]
  resid_weight[i] = math.abs(r["Lbs-"] - pred_w)
  resid_mpg[i] = math.abs(r["Mpg+"] - pred_m)
end

-- Compute mean and std of residuals
local function mean_std(arr)
  local sum, sumsq = 0, 0
  for _, v in ipairs(arr) do
    sum = sum + v
    sumsq = sumsq + v * v
  end
  local n = #arr
  local mean = sum / n
  local std = math.sqrt((sumsq - sum*sum/n)/(n-1))
  return mean, std
end

local mean_w, std_w = mean_std(resid_weight)
local mean_m, std_m = mean_std(resid_mpg)

local thresh_w = mean_w + 2*std_w
local thresh_m = mean_m + 2*std_m

-- Collect suspect rows
local suspect = {}
for i=1,#rows do
  local flag = false
  local diff_w = resid_weight[i]
  local diff_m = resid_mpg[i]
  if diff_w > thresh_w or diff_m > thresh_m then
    flag = true
  end
  if flag then
    table.insert(suspect, {idx=i, diff_w=diff_w, diff_m=diff_m})
  end
end
-- Sort by combined deviation
suspect = sorted(suspect, function(a,b)
  local da = a.diff_w + a.diff_m
  local db = b.diff_w + b.diff_m
  return da > db
end)

local top2 = math.min(5, #suspect)
print(string.format(
  "Idx  DiffW  DiffM  Clndrs  Volume  HpX  Model  origin  Lbs-  Acc+  Mpg+"))
for i=1, top2 do
  local r = rows[suspect[i].idx]
  print(string.format(
    "%3d  %5.1f %5.1f %-7s %-7s %-5s %-6s %-6s %-7s %-5s %-4s",
    suspect[i].idx, suspect[i].diff_w, suspect[i].diff_m,
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"]))
end
print()
