#!/usr/bin/env lua
-- Skill 01: NASA project summary and standout projects
-- Usage: lua skill_01.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_01.lua <csv path>")
end

-- Load all rows
local rows = {}
for r in ezr.csv(path) do
  table.insert(rows, r)
end

-- Simple statistics helpers
local function mean(v)
  local s = 0
  for _, x in ipairs(v) do s = s + x end
  return s / #v
end

local function median(v)
  local s = ezr.copy(v)
  table.sort(s)
  local n = #s
  if n % 2 == 1 then
    return s[(n + 1) / 2]
  else
    return (s[n / 2] + s[n / 2 + 1]) / 2
  end
end

local function sd(v, m)
  local ss = 0
  for _, x in ipairs(v) do ss = ss + (x - m) * (x - m) end
  return math.sqrt(ss / (#v - 1))
end

-- Pull out numeric fields
local ids, size, effort, defects = {}, {}, {}, {}
for _, r in ipairs(rows) do
  table.insert(ids, r[1])                     -- idX
  table.insert(size, r["Kloc+"])
  table.insert(effort, r["Effort-"])
  table.insert(defects, r["Defects-"])
end

-- Summary statistics
local avgSize = mean(size)
local medSize = median(size)
local avgEff  = mean(effort)
local medEff  = median(effort)
local avgDef  = mean(defects)
local medDef  = median(defects)

-- Best / worst defect counts
local minDef, maxDef = math.huge, -math.huge
local minIdx, maxIdx = 1, 1
for i, d in ipairs(defects) do
  if d < minDef then minDef, minIdx = d, i end
  if d > maxDef then maxDef, maxIdx = d, i end
end

-- Compute combined z‑score for each project
local meanSize, meanEff, meanDef = avgSize, avgEff, avgDef
local sdSize, sdEff, sdDef = sd(size, meanSize), sd(effort, meanEff), sd(defects, meanDef)

local scores = {}
for i = 1, #rows do
  local zSize = (size[i] - meanSize) / sdSize
  local zEff  = (effort[i] - meanEff)  / sdEff
  local zDef  = (defects[i] - meanDef) / sdDef
  local score = zSize + zEff + zDef
  table.insert(scores, {
    score = score,
    id    = ids[i],
    K     = size[i],
    E     = effort[i],
    D     = defects[i]
  })
end
table.sort(scores, function(a, b) return a.score < b.score end)

-- Take the top 10% lowest scores (at least one)
local n = #scores
local topCount = math.max(1, math.floor(n * 0.10))
local favorable = {}
for i = 1, topCount do table.insert(favorable, scores[i]) end

-- Output
print("== NASA-01 ==")
print(string.format("Total projects: %d", n))
print(string.format("Avg Kloc+: %s", ezr.round(avgSize)))
print(string.format("Median Kloc+: %s", ezr.round(medSize)))
print(string.format("Avg Effort-: %s", ezr.round(avgEff)))
print(string.format("Median Effort-: %s", ezr.round(medEff)))
print(string.format("Avg Defects-: %s", ezr.round(avgDef)))
print(string.format("Median Defects-: %s", ezr.round(medDef)))
print(string.format(
  "Best outcome (fewest defects): ID %s with %s defects",
  ids[minIdx], ezr.round(defects[minIdx])))
print(string.format(
  "Worst outcome (most defects): ID %s with %s defects",
  ids[maxIdx], ezr.round(defects[maxIdx])))

print("\n== NASA-02 ==")
print("Projects with unusually favorable combinations (top 10% lowest combined z‑score):")
for _, p in ipairs(favorable) do
  print(string.format("ID %s: Kloc+ %.1f, Effort- %.1f, Defects- %.1f",
    p.id, p.K, p.E, p.D))
end
