#!/usr/bin/env lua
--  skill_04.lua – incremental inventory assessment (CAR‑07 & CAR‑08)
--
--  Usage: lua skill_04.lua <csv_path>
--
--  The script first loads the existing CSV, then waits for new
--  vehicle rows on standard input.  For each new row it:
--    * updates the table and all column summaries,
--    * evaluates whether the vehicle would belong in the current
--      top‑N shortlist (default 5),
--    * prints a CAR‑07 report for the new record,
--    * prints an updated CAR‑08 shortlist.
--
--  The script continues until EOF on stdin.  It never
--  restarts the entire analysis – all updates are incremental.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helper functions
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

-- Parse a CSV line into a row (list of values).
local function parse_row(line)
  local row = {}
  for s in (line..","):gmatch("(.-),") do
    table.insert(row, ezr.thing(s))
  end
  return row
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
  io.stderr:write("usage: lua skill_04.lua <csv_path>\n")
  os.exit(1)
end

-- Table with columns and rows
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- Keep handy references to the columns of interest
local cols = {
  Lbs = tbl.cols.all[5],   -- "Lbs-"
  Acc = tbl.cols.all[7],   -- "Acc+"
  Mpg = tbl.cols.all[8]    -- "Mpg+"
}

-- ------------------------------------------------------------------
-- Main loop: read new rows from stdin and process them
-- ------------------------------------------------------------------
local function process_new_row(line)
  local r = parse_row(line)
  -- Skip if the line is empty
  if #r == 0 or (#r == 1 and r[1] == "") then return end

  -- Add to table (updates column summaries)
  tbl:add(r)

  -- Build current top‑N shortlist (default 5)
  local N = 5
  local scored = {}
  for _, rr in ipairs(tbl.rows) do
    local s = composite_score(rr, cols)
    table.insert(scored, {row = rr, score = s})
  end
  table.sort(scored, function(a, b) return a.score > b.score end)
  local top = {}
  for i = 1, math.min(N, #scored) do
    top[i] = scored[i]
  end

  -- Determine if the new record is a candidate
  local is_candidate = false
  for i = 1, #top do
    if top[i].row == r then
      is_candidate = true
      break
    end
  end
  local score_of_new = composite_score(r, cols)

  -- -------------------------------------------------------------
  -- CAR-07: assess the new vehicle
  -- -------------------------------------------------------------
  print("== CAR-07 ==")
  print(string.format("New vehicle: %s", print_row(r)))
  print(string.format("Score: %s", ezr.round(score_of_new, 3)))
  if is_candidate then
    print("Result: Candidate – ranks within top " .. N .. " vehicles.")
  else
    print("Result: Not candidate – outside top " .. N .. ".")
  end
  print()

  -- -------------------------------------------------------------
  -- CAR-08: updated shortlist
  -- -------------------------------------------------------------
  print("== CAR-08 ==")
  print(string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s Score",
    "Clndrs","Vol","HpX","Model","Origin","Lbs-","Acc+","Mpg+"))
  for i = 1, #top do
    local tr = top[i]
    print(string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s %.3f",
      ezr.show(tr.row[1]), ezr.show(tr.row[2]), ezr.show(tr.row[3]), ezr.show(tr.row[4]),
      ezr.show(tr.row[5]), ezr.show(tr.row[6]), ezr.show(tr.row[7]), ezr.show(tr.row[8]),
      ezr.round(tr.score, 3)))
  end
  print()
end

-- Read from stdin until EOF
for line in io.lines() do
  process_new_row(line)
end
