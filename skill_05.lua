#!/usr/bin/env lua
-- Skill 05 – Fuel‑efficiency & overall performance diagnostics
-- Usage:  lua skill_05.lua <csv_path>

local ezr   = require("ezr")
local round = ezr.round
local sorted= ezr.sorted

-- ------------------------------------------------------------------
-- Utility: find column objects by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
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
-- Identify numeric and symbolic columns
local numeric_cols = {}
local symbolic_cols = {}
for _, c in pairs(tbl.cols.all) do
  if type(c.mid) == "function" then -- all columns have mid()
    if c:mid() and c:mid() ~= c:mid() then end -- dummy
  end
  if c.name:find"^%l" then
    table.insert(symbolic_cols, c)
  else
    table.insert(numeric_cols, c)
  end
end

-- ------------------------------------------------------------------
-- Helper math functions
local function mean(arr)
  local sum, n = 0, 0
  for _,v in ipairs(arr) do
    if type(v)=="number" then sum, n = sum+v, n+1 end
  end
  return n>0 and sum/n or nil
end

local function median(arr)
  if #arr==0 then return nil end
  local s = sorted(arr)
  local idx = math.floor(#s/2)+1
  return s[idx]
end

local function mode(arr)
  local cnt = {}
  local bestk, bestn = nil, 0
  for _,v in ipairs(arr) do
    local n = (cnt[v] or 0)+1
    cnt[v] = n
    if n>bestn then bestk, bestn = v, n end
  end
  return bestk
end

local function pearson(x,y)
  local nx,ny = 0,0
  local sx,sy,sxx,syy,sxy = 0,0,0,0,0
  for i=1,#x do
    local a,b = x[i], y[i]
    if type(a)=="number" and type(b)=="number" then
      nx,ny = nx+1,ny+1
      sx,sy = sx+a, sy+b
      sxx,s yy = sxx+a*a, syy+b*b
      sxy = sxy + a*b
    end
  end
  if nx<2 or ny<2 then return nil end
  local denom = math.sqrt((sxx-sx*sx/nx)*(syy-sy*sy/ny))
  if denom==0 then return nil end
  return (sxy - sx*sy/(nx))/denom
end

-- ------------------------------------------------------------------
-- Gather values for each numeric column
local col_vals = {}
for _,c in ipairs(numeric_cols) do
  local vals={}
  for _,row in ipairs(tbl.rows) do
    local v=row[c.at]
    if type(v)=="number" then
      table.insert(vals, v)
    end
  end
  col_vals[c.name] = vals
end

local mpg_vals = col_vals["Mpg+"]
if not mpg_vals or #mpg_vals==0 then
  io.stderr:write("No MPG data available.\n")
  os.exit(1)
end

-- ------------------------------------------------------------------
-- CAR-09: Variables associated with low MPG
print("== CAR-09 ==")

-- 1. Correlations with MPG
local corr_list = {}
for _,c in ipairs(numeric_cols) do
  if c.name ~= "Mpg+" then
    local r = pearson(col_vals[c.name], mpg_vals)
    if r then table.insert(corr_list, {col=c.name, r=r}) end
  end
end
-- sort by absolute correlation descending
table.sort(corr_list, function(a,b)
  return math.abs(a.r) > math.abs(b.r)
end)

for i=1,math.min(3,#corr_list) do
  local item=corr_list[i]
  print(string.format(" %d. %-10s  corr=%.3f",
    i, item.col, item.r))
end

-- 2. Chi‑square for symbolic column (origin) vs low MPG
local origin_col = get_col(tbl, "origin")
local orig_vals = {}
for _,row in ipairs(tbl.rows) do
  local v=row[origin_col.at]
  if v ~= "?" then
    table.insert(orig_vals, v)
  end
end

-- Determine low‑MPG threshold (bottom 20%)
local sorted_mpg = sorted(mpg_vals)
local low_n = math.floor(#sorted_mpg * 0.2) + 1
local threshold = sorted_mpg[low_n]

-- Build contingency table
local total = #orig_vals
local low_count = 0
local orig_total = {}
local orig_low   = {}
for _,v in ipairs(orig_vals) do
  orig_total[v] = (orig_total[v] or 0)+1
end
for _,row in ipairs(tbl.rows) do
  local mpg = row[mpg_col.at]
  local orig = row[origin_col.at]
  if mpg ~= "?" and orig ~= "?" and mpg <= threshold then
    low_count = low_count + 1
    orig_low[orig] = (orig_low[orig] or 0)+1
  end
end

-- chi‑square
local chi = 0
for orig,tot in pairs(orig_total) do
  local expected = low_count * tot / total
  local observed = orig_low[orig] or 0
  if expected > 0 then
    chi = chi + (observed-expected)^2/expected
  end
end
print(string.format("Chi‑square for origin vs low MPG: %.2f", chi))

-- ------------------------------------------------------------------
-- CAR-10: Characteristics of unfavorable overall performance
print("\n== CAR-10 ==")

-- Compute composite score for each row
-- First find min/max for relevant columns
local weight_col = get_col(tbl, "Lbs-")
local acc_col    = get_col(tbl, "Acc+")
local mpg_col    = get_col(tbl, "Mpg+")

local minW, maxW = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge

for _,row in ipairs(tbl.rows) do
  local w=row[weight_col.at]
  local a=row[acc_col.at]
  local m=row[mpg_col.at]
  if type(w)=="number" then minW = math.min(minW,w); maxW = math.max(maxW,w) end
  if type(a)=="number" then minA = math.min(minA,a); maxA = math.max(maxA,a) end
  if type(m)=="number" then minM = math.min(minM,m); maxM = math.max(maxM,m) end
end

local scores = {}
local groups = {fav=[], unfav=[]}
for _,row in ipairs(tbl.rows) do
  local w=row[weight_col.at]
  local a=row[acc_col.at]
  local m=row[mpg_col.at]
  if type(w)=="number" and type(a)=="number" and type(m)=="number" then
    local score =
      (1 - (w-minW)/(maxW-minW)) +      -- lower weight
      (1 - (a-minA)/(maxA-minA)) +      -- lower acc
      ((m-minM)/(maxM-minM))            -- higher mpg
    table.insert(scores, score)
    row._score = score
  end
end

-- Determine median score
local sorted_scores = sorted(scores)
local med = sorted_scores[math.floor(#sorted_scores/2)+1]

-- Split into groups
for _,row in ipairs(tbl.rows) do
  if row._score then
    if row._score < med then
      table.insert(groups.unfav, row)
    else
      table.insert(groups.fav, row)
    end
  end
end

-- Helper to compute group stats
local function group_stats(group)
  local stats = {}
  for _,c in ipairs(numeric_cols) do
    local vals={}
    for _,r in ipairs(group) do
      local v=r[c.at]
      if type(v)=="number" then table.insert(vals,v) end
    end
    stats[c.name] = {mean=mean(vals), vals=vals}
  end
  stats["origin"] = {mode=mode(map(group, function(r) return r[origin_col.at] end))}
  return stats
end

local fav = group_stats(groups.fav)
local unfav = group_stats(groups.unfav)

-- Compute differences
local diffs = {}
for _,c in ipairs(numeric_cols) do
  local a = fav[c.name].mean
  local b = unfav[c.name].mean
  if a and b then
    table.insert(diffs, {col=c.name, fav=a, unfav=b, diff=b-a})
  end
end
table.sort(diffs, function(a,b)
  return math.abs(a.diff) > math.abs(b.diff)
end)

print("Top 3 differences between unfavorable and favorable vehicles:")

for i=1,math.min(3,#diffs) do
  local d = diffs[i]
  print(string.format(" %d. %-10s  Fav=%.1f  Unfav=%.1f  Diff=%.1f",
    i, d.col, d.fav, d.unfav, d.diff))
end

print(string.format("\nCommon origin in unfavorable vehicles: %s",
  unfav.origin.mode))
print(string.format("Common origin in favorable vehicles: %s",
  fav.origin.mode))
