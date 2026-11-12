#!/usr/bin/env lua
-- Skill 08: Advanced vehicle comparison and selection
-- Usage: lua skill_08.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- -----------------------------------------------------------------
-- Utility helpers (copied from previous skills for reuse)
-- -----------------------------------------------------------------
local function col_by_name(tbl, name)
  for _, col in ipairs(tbl.cols.all) do
    if col.name == name then
      return col
    end
  end
  error("column " .. name .. " not found")
end

local function minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _, r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  return minv, maxv
end

local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

-- Composite score used in earlier skills (higher is better)
local function score_for_row(row, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w == "?" or m == "?" or a == "?" then return nil end
  local ws = 1 - norm(w, wmin, wmax)
  local ms = norm(m, mmin, mmax)
  local as = 1 - norm(a, amin, amax)
  return ws + ms + as
end

-- Distance between two rows using all independent columns
local function row_distance(tbl, r1, r2)
  return tbl:distx(r1, r2)
end

-- -----------------------------------------------------------------
-- Load data
-- -----------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_08.lua <csv path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(path))
local rows = tbl.rows

-- Columns used for scoring and comparisons
local wcol = col_by_name(tbl, "Lbs-")
local mcol = col_by_name(tbl, "Mpg+")
local acol = col_by_name(tbl, "Acc+")
local model_col = col_by_name(tbl, "Model")

-- Global min/max for normalisation
local wmin, wmax = minmax(wcol, rows)
local mmin, mmax = minmax(mcol, rows)
local amin, amax = minmax(acol, rows)

-- Compute scores for all rows
local scores = {}
for i, r in ipairs(rows) do
  scores[i] = score_for_row(r, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
end

-- Find the worst scored vehicle (lowest non-nil score)
local worst_i, worst_score = nil, nil
for i, s in ipairs(scores) do
  if s then
    if worst_score == nil or s < worst_score then
      worst_score = s
      worst_i = i
    end
  end
end
local worst_row = rows[worst_i]

-- Compute median score (for CAR-16 threshold)
local score_vals = {}
for _, s in ipairs(scores) do
  if s then table.insert(score_vals, s) end
end
table.sort(score_vals)
local median = score_vals[#score_vals/2 + 1]

-- -----------------------------------------------------------------
-- CAR-15 : alternatives for a poorly rated vehicle
-- -----------------------------------------------------------------
print("== CAR-15 ==")
print(string.format(
  "Target vehicle: Model=%s, Weight=%.0f, MPG=%.1f, Acc=%.1f, Score=%.2f",
  tostring(worst_row[model_col.at]),
  worst_row[wcol.at], worst_row[mcol.at], worst_row[acol.at], round(worst_score,2)))

-- Find neighbors sorted by distance
local neighbors = {}
for i, r in ipairs(rows) do
  if i ~= worst_i then
    local d = row_distance(tbl, worst_row, r)
    table.insert(neighbors, {row=r, dist=d})
  end
end
table.sort(neighbors, function(a,b) return a.dist < b.dist end)

local alternatives = {}
for _, nb in ipairs(neighbors) do
  local s = scores[nb.row[model_col.at] and 1 or nil] -- avoid nil
  if nb.row ~= worst_row and nb.row[model_col.at] and nb.row[model_col.at] ~= "" then
    local idx = table.find(rows, nb.row)
  end
end
-- Build list of alternatives that have a higher score than the target
local alt_candidates = {}
for _, nb in ipairs(neighbors) do
  local idx = nil
  for i,r in ipairs(rows) do
    if r == nb.row then idx = i; break end
  end
  if idx and scores[idx] and scores[idx] > worst_score then
    table.insert(alt_candidates, {row=nb.row, score=scores[idx]})
    if #alt_candidates == 3 then break end
  end
end

if #alt_candidates == 0 then
  print("No better alternatives found.")
else
  print("Top 3 alternatives with better performance:")
  for _, alt in ipairs(alt_candidates) do
    local r = alt.row
    print(string.format(
      "- Model=%s | Weight=%.0f | MPG=%.1f | Acc=%.1f | Score=%.2f",
      tostring(r[model_col.at]),
      r[wcol.at], r[mcol.at], r[acol.at], round(alt.score,2)))
    print(string.format(
      "  ΔWeight=%.0f | ΔMPG=%.1f | ΔAcc=%.1f | ΔScore=%.2f",
      round(r[wcol.at] - worst_row[wcol.at],0),
      round(r[mcol.at] - worst_row[mcol.at],1),
      round(r[acol.at] - worst_row[acol.at],1),
      round(alt.score - worst_score,2)))
  end
end

-- -----------------------------------------------------------------
-- CAR-16 : closest better alternative with minimal changes
-- -----------------------------------------------------------------
print("\n== CAR-16 ==")

-- Pick the first vehicle below the median score
local target_i = nil
for i, s in ipairs(scores) do
  if s and s < median then
    target_i = i
    break
  end
end
if not target_i then
  print("All vehicles meet preferred targets; no alternative needed.")
  os.exit(0)
end
local target_row = rows[target_i]
local target_score = scores[target_i]
print(string.format(
  "Target vehicle (below threshold): Model=%s, Weight=%.0f, MPG=%.1f, Acc=%.1f, Score=%.2f",
  tostring(target_row[model_col.at]),
  target_row[wcol.at], target_row[mcol.at], target_row[acol.at], round(target_score,2)))

-- Find best alternative that is above median and minimal feature change
local best_alt, best_delta, best_score = nil, math.huge, nil
for i, r in ipairs(rows) do
  local s = scores[i]
  if s and s >= median and r ~= target_row then
    local delta =
      math.abs(r[wcol.at] - target_row[wcol.at]) +
      math.abs(r[mcol.at] - target_row[mcol.at]) +
      math.abs(r[acol.at] - target_row[acol.at])
    if delta < best_delta or (delta == best_delta and s > best_score) then
      best_alt = r
      best_delta = delta
      best_score = s
    end
  end
end

if not best_alt then
  print("No better alternative found.")
else
  print(string.format(
    "Closest better alternative: Model=%s, Weight=%.0f, MPG=%.1f, Acc=%.1f, Score=%.2f",
    tostring(best_alt[model_col.at]),
    best_alt[wcol.at], best_alt[mcol.at], best_alt[acol.at], round(best_score,2)))
  print(string.format(
    "Feature changes: ΔWeight=%.0f, ΔMPG=%.1f, ΔAcc=%.1f | Score improvement=%.2f",
    round(best_alt[wcol.at] - target_row[wcol.at],0),
    round(best_alt[mcol.at] - target_row[mcol.at],1),
    round(best_alt[acol.at] - target_row[acol.at],1),
    round(best_score - target_score,2)))
end
