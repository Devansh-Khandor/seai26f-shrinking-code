#!/usr/bin/env lua
-- Skill 01 – Inventory summary and top‑combination vehicles
-- Usage:  lua skill_01.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper functions
-- ------------------------------------------------------------------
local function col_values(tbl, name)
  local col = tbl.cols.all[name]
  local vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[col.at]
    if v ~= "?" then table.insert(vals, v) end
  end
  return vals, col
end

local function mean(vs)          -- arithmetic mean
  local sum = 0
  for _, v in ipairs(vs) do sum = sum + v end
  return sum / #vs
end

local function min(vs)           -- min of list
  local m = vs[1]
  for _, v in ipairs(vs) do if v < m then m = v end end
  return m
end

local function max(vs)           -- max of list
  local m = vs[1]
  for _, v in ipairs(vs) do if v > m then m = v end end
  return m
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

local weights, weight_col = col_values(tbl, "Lbs-")
local mpgs, mpg_col       = col_values(tbl, "Mpg+")
local accs, acc_col       = col_values(tbl, "Acc+")

local w_mean = mean(weights)
local m_mean = mean(mpgs)
local a_mean = mean(accs)

local w_min = min(weights)
local w_max = max(weights)
local m_min = min(mpgs)
local m_max = max(mpgs)
local a_min = min(accs)
local a_max = max(accs)

-- -------------------------------------------------------------
-- CAR-01 : Inventory summary
-- -------------------------------------------------------------
print("== CAR-01 ==")
print(string.format("Typical weight: %.2f lb", w_mean))
print(string.format("Typical mpg   : %.2f", m_mean))
print(string.format("Typical acc   : %.2f", a_mean))
print(string.format("Best weight   : %.2f lb", w_min))
print(string.format("Worst weight  : %.2f lb", w_max))
print(string.format("Best mpg      : %.2f", m_max))
print(string.format("Worst mpg     : %.2f", m_min))
print(string.format("Best acc      : %.2f", a_min))
print(string.format("Worst acc     : %.2f", a_max))

-- -------------------------------------------------------------
-- CAR-02 : Most attractive vehicles
-- -------------------------------------------------------------
local function score_row(r)
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]
  local w_s = (w_max - w) / (w_max - w_min)   -- low weight better
  local m_s = (m - m_min)   / (m_max - m_min)   -- high mpg better
  local a_s = (a_max - a)   / (a_max - a_min)   -- low acc better
  return w_s + m_s + a_s
end

local scored = {}
for _, r in ipairs(tbl.rows) do
  local s = score_row(r)
  table.insert(scored, {row = r, score = s})
end

table.sort(scored, function(a, b) return a.score > b.score end)

print("\n== CAR-02 ==")
print("Top vehicles by combined attractiveness (weight, mpg, acceleration):")
for i = 1, math.min(5, #scored) do
  local entry = scored[i]
  print(string.format("%2d) score=%.4f %s",
          i, entry.score, ezr.show(entry.row)))
end
