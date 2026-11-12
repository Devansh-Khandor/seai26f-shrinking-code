#!/usr/bin/env lua
-- Skill 08 – Alternative vehicle recommendations
-- Usage:  lua skill_08.lua <csv path>

local ezr = require "ezr"

----------------------------------------------------------------
-- Helper utilities (mostly copied from previous skills)
----------------------------------------------------------------
local function mean(vs)
  local sum, n = 0, 0
  for _, v in ipairs(vs) do
    if v ~= "?" then sum, n = sum + v, n + 1 end
  end
  return n > 0 and sum / n or nil
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

-- Score based on weight, mpg and acceleration
local function score_row(r, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]
  local w_s = (w_max - w) / (w_max - w_min)          -- low weight better
  local m_s = (m - m_min)   / (m_max - m_min)        -- high mpg better
  local a_s = (a_max - a)   / (a_max - a_min)        -- low acc better
  return w_s + m_s + a_s
end

----------------------------------------------------------------
-- Main
----------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- Relevant columns
local weight_col = tbl.cols.all["Lbs-"]
local mpg_col    = tbl.cols.all["Mpg+"]
local acc_col    = tbl.cols.all["Acc+"]
local weight_vals = {}
local mpg_vals    = {}
local acc_vals    = {}
for _, r in ipairs(tbl.rows) do
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]
  if w ~= "?" then table.insert(weight_vals, w) end
  if m ~= "?" then table.insert(mpg_vals, m)   end
  if a ~= "?" then table.insert(acc_vals, a)   end
end

-- Summary stats
local w_mean, w_min, w_max = mean(weight_vals), min(weight_vals), max(weight_vals)
local m_mean, m_min, m_max = mean(mpg_vals),    min(mpg_vals),    max(mpg_vals)
local a_mean, a_min, a_max = mean(acc_vals),    min(acc_vals),    max(acc_vals)

-- Compute score for each row
local scored = {}
for _, r in ipairs(tbl.rows) do
  local s = score_row(r, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)
  table.insert(scored, {row = r, score = s})
end

----------------------------------------------------------------
-- CAR-15 : Alternative vehicles for a poorly rated car
----------------------------------------------------------------
print("\n== CAR-15 ==")

-- Find the worst scored vehicle
table.sort(scored, function(a,b) return a.score < b.score end)
local worst = scored[1].row
local worst_score = scored[1].score

print(string.format("Worst vehicle (score=%.4f): %s", worst_score, ezr.show(worst)))

-- Compute distances from worst to all others
local others = {}
for _, r in ipairs(tbl.rows) do
  if r ~= worst then
    local d = tbl:distx(worst, r)
    table.insert(others, {row = r, dist = d, score = score_row(r, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)})
  end
end
-- Sort nearest
table.sort(others, function(a,b) return a.dist < b.dist end)

-- Pick first 3 alternatives with better score
local alternatives = {}
for _, o in ipairs(others) do
  if o.score > worst_score then
    table.insert(alternatives, o)
    if #alternatives >= 3 then break end
  end
end

for i, alt in ipairs(alternatives) do
  local d = alt.dist
  local w = alt.row[weight_col.at]
  local m = alt.row[mpg_col.at]
  local a = alt.row[acc_col.at]
  print(string.format("\nAlternative %d (dist=%.4f, score=%.4f): %s", i, d, alt.score, ezr.show(alt.row)))
  print(string.format("  Weight improved by %.1f lb", worst[weight_col.at] - w))
  print(string.format("  MPG improved by %.1f", m - worst[mpg_col.at]))
  print(string.format("  Acceleration improved by %.1f", worst[acc_col.at] - a))
end

----------------------------------------------------------------
-- CAR-16 : Closest vehicle meeting more performance targets
----------------------------------------------------------------
print("\n== CAR-16 ==")

-- Define preferred targets (lower weight, higher mpg, lower acceleration)
local function target_met(r)
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]
  local count = 0
  if w ~= "?" and w < w_mean then count = count + 1 end
  if m ~= "?" and m > m_mean then count = count + 1 end
  if a ~= "?" and a < a_mean then count = count + 1 end
  return count
end

-- Find a vehicle that does not meet all targets
local candidates = {}
for _, r in ipairs(tbl.rows) do
  if target_met(r) < 3 then
    table.insert(candidates, r)
  end
end
if #candidates == 0 then
  print("All vehicles meet all preferred targets. Nothing to recommend.")
  return
end

-- Pick the vehicle with the fewest targets met
table.sort(candidates, function(a,b) return target_met(a) < target_met(b) end)
local problematic = candidates[1]
local problematic_score = score_row(problematic, weight_col, mpg_col, acc_col, w_min, w_max, m_min, m_max, a_min, a_max)

print(string.format("Problematic vehicle (targets met=%d): %s", target_met(problematic), ezr.show(problematic)))

-- Find best alternative: maximize targets met, then minimize distance
local best_alt = nil
local best_targets = -1
local best_dist = nil
for _, r in ipairs(tbl.rows) do
  if r ~= problematic then
    local t = target_met(r)
    local d = tbl:distx(problematic, r)
    if t > best_targets or (t == best_targets and (best_dist == nil or d < best_dist)) then
      best_alt = r
      best_targets = t
      best_dist = d
    end
  end
end

local alt_w = best_alt[weight_col.at]
local alt_m = best_alt[mpg_col.at]
local alt_a = best_alt[acc_col.at]

print(string.format("\nRecommended alternative (targets met=%d, dist=%.4f): %s", best_targets, best_dist, ezr.show(best_alt)))
print(string.format("  Weight change: %.1f lb (%s vs %.1f)", alt_w - problematic[weight_col.at],
      alt_w < problematic[weight_col.at] and "lower" or "higher", problematic[weight_col.at]))
print(string.format("  MPG change: %.1f (%s vs %.1f)", alt_m - problematic[mpg_col.at],
      alt_m > problematic[mpg_col.at] and "higher" or "lower", problematic[mpg_col.at]))
print(string.format("  Acceleration change: %.1f (%s vs %.1f)", alt_a - problematic[acc_col.at],
      alt_a < problematic[acc_col.at] and "lower" or "higher", problematic[acc_col.at]))
print(string.format("  This vehicle improves %s of the preferred performance targets compared to the problematic one.",
      best_targets == 3 and "all" or best_targets == 2 and "two" or best_targets == 1 and "one" or "none"))
