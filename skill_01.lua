#!/usr/bin/env lua
-- Skill 01: NASA project summary & exceptional projects finder
-- Usage: lua skill_01.lua <csv_path>

local ezr = require("ezr-lib")      -- library that provides csv, etc.
local csv = ezr.csv

-- ---------- Utility functions ------------------------------------
local function map(t, f)
  local r = {}
  for _, v in ipairs(t) do r[#r+1] = f(v) end
  return r
end

local function filter(t, pred)
  local r = {}
  for _, v in ipairs(t) do
    if pred(v) then r[#r+1] = v end
  end
  return r
end

local function stats(values)
  local n = #values
  local sum, minv, maxv = 0, math.huge, -math.huge
  for _, v in ipairs(values) do
    sum = sum + v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local sumsq = 0
  for _, v in ipairs(values) do
    sumsq = sumsq + (v - mean) * (v - mean)
  end
  local sd = math.sqrt(sumsq / n)
  return {n = n, mean = mean, min = minv, max = maxv, sd = sd}
end

-- ---------- Read CSV and collect relevant data -------------------
local file = arg[1]
if not file then
  io.stderr:write("Usage: lua skill_01.lua <csv_path>\n")
  os.exit(1)
end

local rows = {}
local header = nil
local idx = {}

for row in csv(file) do
  if not header then
    header = row
    for i, name in ipairs(row) do idx[name] = i end
  else
    local rec = {}
    rec.id = tostring(row[idx["idX"]]) or tostring(row[1])
    rec.Kloc = tonumber(row[idx["Kloc+"]])
    rec.Effort = tonumber(row[idx["Effort-"]])
    rec.Defects = tonumber(row[idx["Defects-"]])
    rec.Months = tonumber(row[idx["MonthsX"]])
    if rec.Kloc and rec.Effort and rec.Defects then
      table.insert(rows, rec)
    end
  end
end

-- ---------- Compute statistics ------------------------------------
local Klocs   = map(rows, function(r) return r.Kloc end)
local Efforts = map(rows, function(r) return r.Effort end)
local Defects = map(rows, function(r) return r.Defects end)

local statsKloc   = stats(Klocs)
local statsEffort = stats(Efforts)
local statsDefects = stats(Defects)

-- Best / worst by combined Effort + Defects
local function outcome(r) return r.Effort + r.Defects end
local best, worst
local bestScore, worstScore = math.huge, -math.huge
for _, r in ipairs(rows) do
  local s = outcome(r)
  if s < bestScore then bestScore, best = s, r end
  if s > worstScore then worstScore, worst = s, r end
end

-- ---------- Find unusually favorable projects ---------------------
local favorable = {}
local effThresh = statsEffort.mean - statsEffort.sd
local defThresh = statsDefects.mean - statsDefects.sd
local locThresh = statsKloc.mean + statsKloc.sd

for _, r in ipairs(rows) do
  if r.Effort < effThresh and r.Defects < defThresh and r.Kloc > locThresh then
    table.insert(favorable, r)
  end
end

-- ---------- Output -----------------------------------------------
local function fmt_num(v)
  if math.floor(v) == v then return tostring(math.floor(v)) end
  return string.format("%.2f", v)
end

print("== NASA-01 ==")
print(string.format("Typical size (Kloc+): mean=%s, sd=%s, min=%s, max=%s",
  fmt_num(statsKloc.mean), fmt_num(statsKloc.sd),
  fmt_num(statsKloc.min), fmt_num(statsKloc.max)))
print(string.format("Typical effort (Effort-): mean=%s, sd=%s, min=%s, max=%s",
  fmt_num(statsEffort.mean), fmt_num(statsEffort.sd),
  fmt_num(statsEffort.min), fmt_num(statsEffort.max)))
print(string.format("Typical defects (Defects-): mean=%s, sd=%s, min=%s, max=%s",
  fmt_num(statsDefects.mean), fmt_num(statsDefects.sd),
  fmt_num(statsDefects.min), fmt_num(statsDefects.max)))
print(string.format("Best outcome: ID=%s, Effort=%s, Defects=%s",
  best.id, fmt_num(best.Effort), fmt_num(best.Defects)))
print(string.format("Worst outcome: ID=%s, Effort=%s, Defects=%s",
  worst.id, fmt_num(worst.Effort), fmt_num(worst.Defects)))

print("\n== NASA-02 ==")
print("Projects with unusually favorable combination (Effort & Defects below mean - SD, size above mean + SD):")
if #favorable == 0 then
  print("None found.")
else
  for _, r in ipairs(favorable) do
    print(string.format("ID=%s: Effort=%s, Defects=%s, Kloc+%s",
      r.id, fmt_num(r.Effort), fmt_num(r.Defects), fmt_num(r.Kloc)))
  end
end
