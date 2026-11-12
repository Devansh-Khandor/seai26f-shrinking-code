#!/usr/bin/env lua
-- Skill 04 – Real‑time inventory assessment and dynamic summaries
-- Usage:  lua skill_04.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper: fetch column values and the column itself
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

-- ------------------------------------------------------------------
-- Helper: simple statistics
-- ------------------------------------------------------------------
local function mean(vs)
  local sum = 0
  for _, v in ipairs(vs) do sum = sum + v end
  return #vs > 0 and sum / #vs or nil
end

local function min(vs)
  local m = vs[1]
  for _, v in ipairs(vs) do
    if v < m then m = v end
  end
  return m
end

local function max(vs)
  local m = vs[1]
  for _, v in ipairs(vs) do
    if v > m then m = v end
  end
  return m
end

-- ------------------------------------------------------------------
-- Scoring function (same as CAR‑02)
-- ------------------------------------------------------------------
local function score_row(r, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]
  local w_s = (w_max - w) / (w_max - w_min)          -- low weight better
  local m_s = (m - m_min)   / (m_max - m_min)        -- high mpg better
  local a_s = (a_max - a)   / (a_max - a_min)        -- low acc better
  return w_s + m_s + a_s
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

-- Extract relevant columns
local weights, weight_col = col_values(tbl, "Lbs-")
local mpgs, mpg_col       = col_values(tbl, "Mpg+")
local accs, acc_col       = col_values(tbl, "Acc+")

-- Compute summary statistics
local w_mean = mean(weights)
local m_mean = mean(mpgs)
local a_mean = mean(accs)

local w_min = min(weights)
local w_max = max(weights)
local m_min = min(mpgs)
local m_max = max(mpgs)
local a_min = min(accs)
local a_max = max(accs)

-- Compute scores for every row
local scored = {}
for _, r in ipairs(tbl.rows) do
  local s = score_row(r, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)
  table.insert(scored, {row = r, score = s})
end
table.sort(scored, function(a, b) return a.score > b.score end)

-- ------------------------------------------------------------------
-- CAR‑07 : Immediate assessment of each vehicle
-- ------------------------------------------------------------------
print("== CAR-07 ==")
print("Top 5 vehicles by combined attractiveness (weight, mpg, acceleration):")
for i = 1, math.min(5, #scored) do
  local entry = scored[i]
  print(string.format("%2d) score=%.4f %s",
          i, entry.score, ezr.show(entry.row)))
end
print("") -- spacer

-- ------------------------------------------------------------------
-- CAR‑08 : Up‑to‑date summary and recommended shortlist
-- ------------------------------------------------------------------
print("== CAR-08 ==")
print("Inventory summary:")
print(string.format("Typical weight: %.2f lb", w_mean))
print(string.format("Typical mpg   : %.2f",   m_mean))
print(string.format("Typical acc   : %.2f",   a_mean))
print(string.format("Best weight   : %.2f lb", w_min))
print(string.format("Worst weight  : %.2f lb", w_max))
print(string.format("Best mpg      : %.2f",   m_max))
print(string.format("Worst mpg     : %.2f",   m_min))
print(string.format("Best acc      : %.2f",   a_min))
print(string.format("Worst acc     : %.2f",   a_max))
print("\nRecommended top 5 vehicles:")
for i = 1, math.min(5, #scored) do
  local entry = scored[i]
  print(string.format("%2d) score=%.4f %s",
          i, entry.score, ezr.show(entry.row)))
end
