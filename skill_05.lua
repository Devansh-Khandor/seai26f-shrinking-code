#!/usr/bin/env lua
-- Skill 05: NASA-09 & NASA-10
-- Usage: lua skill_05.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_05.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Read CSV – header + rows
-- ------------------------------------------------------------------
local rows, header = {}, nil
for r in ezr.csv(path) do
  if not header then header = r else table.insert(rows, r) end
end

-- ------------------------------------------------------------------
-- Column indices
-- ------------------------------------------------------------------
local idx = {}
for i, name in ipairs(header) do idx[name] = i end

local effort_col  = idx["Effort-"]
local defects_col = idx["Defects-"]
local id_col      = idx["idX"]

if not (effort_col and defects_col and id_col) then
  error("Required columns not found in the CSV")
end

-- ------------------------------------------------------------------
-- Helper stats
-- ------------------------------------------------------------------
local function mean(t)
  local s = 0
  for _, v in ipairs(t) do s = s + v end
  return s / #t end

local function sd(t, m)
  local ss = 0
  for _, v in ipairs(t) do ss = ss + (v - m) ^ 2 end
  return math.sqrt(ss / (#t - 1)) end

-- ------------------------------------------------------------------
-- Identify numeric columns (first row determines type)
-- ------------------------------------------------------------------
local numeric = {}
local first = rows[1]
for i, name in ipairs(header) do
  if type(first[i]) == "number" then
    numeric[name] = i
  end
end

-- ------------------------------------------------------------------
-- Extract target arrays
-- ------------------------------------------------------------------
local effort  = {}
local defects = {}
for _, r in ipairs(rows) do
  table.insert(effort,  r[effort_col])
  table.insert(defects, r[defects_col])
end

local meanEff  = mean(effort)
local sdEff    = sd(effort, meanEff)
local meanDef  = mean(defects)
local sdDef    = sd(defects, meanDef)

-- ------------------------------------------------------------------
-- Correlation helper
-- ------------------------------------------------------------------
local function corr(x, y, mx, my, sx, sy)
  if sx == 0 or sy == 0 then return nil end
  local n = #x
  local cov = 0
  for i = 1, n do
    cov = cov + (x[i] - mx) * (y[i] - my)
  end
  cov = cov / n
  return cov / (sx * sy)
end

-- ------------------------------------------------------------------
-- Compute correlations for each numeric column
-- ------------------------------------------------------------------
local corrEff = {}
local corrDef = {}

for name, at in pairs(numeric) do
  if name ~= "Effort-" and name ~= "Defects-" and name ~= "idX" then
    local col = {}
    for _, r in ipairs(rows) do table.insert(col, r[at]) end
    local mx = mean(col)
    local sx = sd(col, mx)

    -- Correlation with effort
    local ce = corr(col, effort, mx, meanEff, sx, sdEff)
    if ce then corrEff[name] = ce end

    -- Correlation with defects
    local cd = corr(col, defects, mx, meanDef, sx, sdDef)
    if cd then corrDef[name] = cd end
  end
end

-- ------------------------------------------------------------------
-- Helper to pick top N by absolute correlation
-- ------------------------------------------------------------------
local function top_n(cor, n)
  local arr = {}
  for k, v in pairs(cor) do arr[#arr + 1] = {k = k, v = v} end
  table.sort(arr, function(a, b) return math.abs(b.v) < math.abs(a.v) end)
  if #arr > n then
    local res = {}
    for i = 1, n do res[i] = arr[i] end
    return res
  else
    return arr
  end
end

local topEff = top_n(corrEff, 3)
local topDef = top_n(corrDef, 3)

-- ------------------------------------------------------------------
-- Output
-- ------------------------------------------------------------------
print("== NASA-09 ==")
print("\nTop attributes correlated with high effort:")
for _, a in ipairs(topEff) do
  print(string.format("  %s: %+.3f", a.k, ezr.round(a.v, 3)))
end

print("\n== NASA-10 ==")
print("\nTop attributes correlated with high defects:")
for _, a in ipairs(topDef) do
  print(string.format("  %s: %+.3f", a.k, ezr.round(a.v, 3)))
end
