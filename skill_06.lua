#!/usr/bin/env lua
--  skill_06.lua – vehicle recommendation explanations (CAR‑11 & CAR‑12)
--  Usage: lua skill_06.lua <csv_path>
--
--  Depends on the ezr library (ezr.lua & ezr-lib.lua) in the
--  same directory or in the Lua package path.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
-- Normalised score for a column.
local function norm_score(col, r, maximize)
  local v = r[col.at]
  if v == "?" then return 0.5 end        -- neutral if unknown
  local n = col:norm(v)
  return maximize and n or (1 - n)
end

-- Composite desirability score: higher is better.
local function composite_score(r, cols)
  return (norm_score(cols.Lbs, r, false) +
          norm_score(cols.Acc, r, true) +
          norm_score(cols.Mpg, r, true)) / 3
end

-- Pretty‑print a row of the table.
local function print_row(r)
  return string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s",
    ezr.show(r[1]), ezr.show(r[2]), ezr.show(r[3]), ezr.show(r[4]),
    ezr.show(r[5]), ezr.show(r[6]), ezr.show(r[7]), ezr.show(r[8]))
end

-- ------------------------------------------------------------------
-- Load existing data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_06.lua <csv_path>\n")
  os.exit(1)
end

-- Build the table.
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- Reference to the columns we care about.
local cols = {
  Lbs = tbl.cols.all[5],   -- "Lbs-"
  Acc = tbl.cols.all[7],   -- "Acc+"
  Mpg = tbl.cols.all[8]    -- "Mpg+"
}

-- ------------------------------------------------------------------
-- Build top‑N shortlist (default 5) – same as in skill_04
-- ------------------------------------------------------------------
local function build_shortlist()
  local scored = {}
  for _, r in ipairs(tbl.rows) do
    local s = composite_score(r, cols)
    table.insert(scored, {row = r, score = s})
  end
  table.sort(scored, function(a,b) return a.score > b.score end)
  local top = {}
  for i = 1, math.min(5, #scored) do top[i] = scored[i] end
  return top
end

-- ------------------------------------------------------------------
-- CAR‑11: Recommendation explanation
-- ------------------------------------------------------------------
print("== CAR-11 ==")
local top = build_shortlist()
for i, t in ipairs(top) do
  local r = t.row
  local s = t.score
  local w = r[cols.Lbs.at]
  local a = r[cols.Acc.at]
  local m = r[cols.Mpg.at]
  local nw = norm_score(cols.Lbs, r, false)
  local na = norm_score(cols.Acc, r, true)
  local nm = norm_score(cols.Mpg, r, true)

  print(string.format(
    "%d) Vehicle [%s] Score %.3f: weight %.2f (norm %.2f), acc %.2f (norm %.2f), mpg %.2f (norm %.2f)",
    i, print_row(r), ezr.round(s,3), w, ezr.round(nw,3), a, ezr.round(na,3), m, ezr.round(nm,3)))

  if i < #top then
    local r_next = top[i+1].row
    local nw_next = norm_score(cols.Lbs, r_next, false)
    local na_next = norm_score(cols.Acc, r_next, true)
    local nm_next = norm_score(cols.Mpg, r_next, true)

    print(string.format(
      "   Compared to vehicle %d: weight diff %.2f (%.2f), acc diff %.2f (%.2f), mpg diff %.2f (%.2f)",
      i+1,
      w - r_next[cols.Lbs.at], ezr.round(nw - nw_next,3),
      a - r_next[cols.Acc.at], ezr.round(na - na_next,3),
      m - r_next[cols.Mpg.at], ezr.round(nm - nm_next,3)))
  end
  print()
end

-- ------------------------------------------------------------------
-- CAR‑12: Trade‑off comparison of two arbitrary vehicles
-- ------------------------------------------------------------------
print("== CAR-12 ==")
io.write("Enter two row numbers (1‑based) separated by space: ")
local line = io.read()
if not line then
  print("No input received. Exiting.")
  os.exit(0)
end
local a_str, b_str = line:match("(%d+)%s+(%d+)")
if not a_str then
  print("Invalid format. Expected two integers.")
  os.exit(1)
end
local a, b = tonumber(a_str), tonumber(b_str)
if a < 1 or a > #tbl.rows or b < 1 or b > #tbl.rows then
  print("Row numbers out of range.")
  os.exit(1)
end

local r1, r2 = tbl.rows[a], tbl.rows[b]
local s1, s2 = composite_score(r1, cols), composite_score(r2, cols)

print(string.format("Vehicle %d score: %.3f", a, ezr.round(s1,3)))
print(string.format("Vehicle %d score: %.3f", b, ezr.round(s2,3)))

local better = s1 > s2
print(string.format("Vehicle %s receives higher overall recommendation.",
  better and a or b))

print("\nTrade‑offs:")
print(string.format("  Weight difference: %.2f (positive => first heavier)",
  r1[cols.Lbs.at] - r2[cols.Lbs.at]))
print(string.format("  Acceleration difference: %.2f (positive => first faster)",
  r1[cols.Acc.at] - r2[cols.Acc.at]))
print(string.format("  MPG difference: %.2f (positive => first higher)",
  r1[cols.Mpg.at] - r2[cols.Mpg.at]))
