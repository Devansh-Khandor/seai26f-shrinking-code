#!/usr/bin/env lua
-- Skill 04: Incremental assessment and dynamic updates
-- Usage: lua skill_04.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- -----------------------------------------------------------------
-- Helper functions
-- -----------------------------------------------------------------
local function col_by_name(cols, name)
  for _,c in ipairs(cols.all) do
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

local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

local function score_for_row(row, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w == "?" or m == "?" or a == "?" then return nil end
  local ws = 1 - norm(w, wmin, wmax)   -- lower weight better
  local ms = norm(m, mmin, mmax)       -- higher mpg better
  local as = 1 - norm(a, amin, amax)   -- lower acceleration better
  return ws + ms + as -- 0 .. 3
end

-- Return indices of top N rows by score (descending)
local function top_indices(rows, scores, n)
  local idx = {}
  for i = 1, #rows do idx[i] = i end
  table.sort(idx, function(i,j)
    local si, sj = scores[i] or -math.huge, scores[j] or -math.huge
    return si > sj
  end)
  local top = {}
  for i = 1, math.min(n, #idx) do
    top[i] = idx[i]
  end
  return top
end

-- -----------------------------------------------------------------
-- Main program
-- -----------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_04.lua <csv path>\n")
  os.exit(1)
end

local csv_iter = ezr.csv(path)
local header = csv_iter()
if not header then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

local cols = ezr.Cols(header)

-- Column objects used in scoring
local wcol = col_by_name(cols, "Lbs-")
local mcol = col_by_name(cols, "Mpg+")
local acol = col_by_name(cols, "Acc+")
local model_col = col_by_name(cols, "Model")

-- Storage for rows processed so far
local rows = {}

-- -----------------------------------------------------------------
-- CAR-07 : assess each new vehicle immediately
-- -----------------------------------------------------------------
print("== CAR-07 ==")

while true do
  local row = csv_iter()
  if not row then break end
  table.insert(rows, row)
  cols:add(row)

  -- Current global stats
  local wmin, wmax = minmax(wcol, rows)
  local mmin, mmax = minmax(mcol, rows)
  local amin, amax = minmax(acol, rows)

  -- Score for this row
  local s = score_for_row(row, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)

  -- Build scores for all rows to find top 5
  local scores = {}
  for i,r in ipairs(rows) do
    scores[i] = score_for_row(r, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
  end
  local top = top_indices(rows, scores, 5)
  local in_top = false
  for _,i in ipairs(top) do
    if rows[i] == row then in_top = true break end
  end

  local model_val = row[model_col.at]
  if s then
    print(string.format("Vehicle %s score %.2f; top5? %s",
          tostring(model_val), round(s,2), in_top and "YES" or "NO"))
  else
    print(string.format("Vehicle %s has missing data; top5? N/A",
          tostring(model_val)))
  end
end

-- -----------------------------------------------------------------
-- CAR-08 : final summaries and recommended shortlist
-- -----------------------------------------------------------------
print("\n== CAR-08 ==")

-- Final stats
local wmin, wmax = minmax(wcol, rows)
local mmin, mmax = minmax(mcol, rows)
local amin, amax = minmax(acol, rows)

print(string.format(
  "Typical weight: %.0f (± %.0f), best: %.0f, worst: %.0f",
  round(wcol:mid()), round(wcol:div()), wmin, wmax))

print(string.format(
  "Typical mpg: %.1f (± %.1f), best: %.1f, worst: %.1f",
  round(mcol:mid()), round(mcol:div()), mmax, mmin))

print(string.format(
  "Typical acceleration: %.1f (± %.1f), best: %.1f, worst: %.1f",
  round(acol:mid()), round(acol:div()), amin, amax))

-- Scores for all rows
local scores = {}
for i,r in ipairs(rows) do
  scores[i] = score_for_row(r, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
end

local top = top_indices(rows, scores, 5)

print(string.format("%-10s %-6s %-6s %-6s %-6s",
      "Model","Weight","MPG","Acc","Score"))
for _,i in ipairs(top) do
  local r = rows[i]
  local s = scores[i] or 0
  print(string.format("%-10s %-6s %-6s %-6s %-6.2f",
        tostring(r[model_col.at]), tostring(r[wcol.at]),
        tostring(r[mcol.at]), tostring(r[acol.at]), round(s,2)))
end
