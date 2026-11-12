#!/usr/bin/env lua
-- Skill 01: Summary and top vehicles
-- Usage: lua skill_01.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- Helper: find a column object by its name
local function col_by_name(tbl, name)
  for at, col in ipairs(tbl.cols.all) do
    if col.name == name then return col end
  end
  error("column " .. name .. " not found")
end

-- Helper: compute min/max of a column
local function minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _, row in ipairs(rows) do
    local v = row[col.at]
    if v ~= "?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  return minv, maxv
end

-- Read CSV and build the table
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_01.lua <csv path>\n")
  os.exit(1)
end
local tbl = ezr.Tbl(ezr.csv(path))

-- ---- CAR-01 ----
print("== CAR-01 ==")

local wcol   = col_by_name(tbl, "Lbs-")
local mcol   = col_by_name(tbl, "Mpg+")
local acol   = col_by_name(tbl, "Acc+")

local wmin, wmax = minmax(wcol, tbl.rows)
local mmin, mmax = minmax(mcol, tbl.rows)
local amin, amax = minmax(acol, tbl.rows)

print(string.format(
  "Typical weight: %.0f (± %.0f), best: %.0f, worst: %.0f",
  round(wcol:mid()), round(wcol:div()), wmin, wmax))

print(string.format(
  "Typical mpg: %.1f (± %.1f), best: %.1f, worst: %.1f",
  round(mcol:mid()), round(mcol:div()), mmax, mmin))

print(string.format(
  "Typical acceleration: %.1f (± %.1f), best: %.1f, worst: %.1f",
  round(acol:mid()), round(acol:div()), amin, amax))

-- ---- CAR-02 ----
print("\n== CAR-02 ==")

-- Normalization helpers
local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end -- avoid div0
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

-- Compute scores for each row
local scores = {}
for _, row in ipairs(tbl.rows) do
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w ~= "?" and m ~= "?" and a ~= "?" then
    local ws = 1 - norm(w, wmin, wmax)   -- lower weight better
    local ms = norm(m, mmin, mmax)       -- higher mpg better
    local as = 1 - norm(a, amin, amax)   -- lower accel better
    local score = ws + ms + as            -- 0..3
    table.insert(scores, {row=row, score=score})
  end
end

-- Sort by score descending
table.sort(scores, function(a,b) return a.score > b.score end)

-- Pick top 5 (or fewer if dataset small)
local top = {}
for i=1, math.min(5, #scores) do
  table.insert(top, scores[i])
end

-- Print top vehicles
if #top == 0 then
  print("No complete data to rank vehicles.")
else
  local mcolat = mcol.at
  local wcolat = wcol.at
  local acolat = acol.at
  local modcol = col_by_name(tbl, "Model")
  local modat = modcol.at
  print(string.format("%-10s %-6s %-6s %-6s %-6s",
        "Model","Weight","MPG","Acc","Score"))
  for _, item in ipairs(top) do
    local r = item.row
    print(string.format("%-10s %-6s %-6s %-6s %-6.2f",
          tostring(r[modat] or "?"),
          tostring(r[wcolat]),
          tostring(r[mcolat]),
          tostring(r[acolat]),
          round(item.score,2)))
  end
end
