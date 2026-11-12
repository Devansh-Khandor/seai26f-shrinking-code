#!/usr/bin/env lua
-- skill_07.lua
-- APACHE-13 & APACHE-14: low‑performance discovery & early‑evaluation strategy
--
-- Usage: lua skill_07.lua <csv_path> [budget]
--   budget – optional, number of rows to evaluate in APACHE‑14 (default 10)
--
-- Requires only the ezr library (ezr.lua & ezr-lib.lua) in the same directory.
--
-- The script prints two sections:
--   == APACHE-13 ==   (lowest‑performance configuration)
--   == APACHE-14 ==   (suggested evaluation order for a given budget)

local ezr = require("ezr")
local csv  = ezr.csv
local TBL  = ezr.Tbl
local Tree = ezr.Tree
local sorted = ezr.sorted
local keysort = ezr.keysort
local rand = ezr.rand
local copy = ezr.copy
local show = ezr.show

-- ------------------------------------------------------------------
-- Load the CSV file as a TBL
-- ------------------------------------------------------------------
local function load_tbl(path)
  return TBL(csv(path))
end

-- ------------------------------------------------------------------
-- Find the row with the minimum performance value
-- ------------------------------------------------------------------
local function find_best(tbl)
  local best = nil
  local best_i = nil
  for i, r in ipairs(tbl.rows) do
    local v = r[tbl.cols.y[1].at]
    if v ~= "?" then
      if not best or v < best then
        best, best_i = v, i
      end
    end
  end
  return best_i, best
end

-- ------------------------------------------------------------------
-- Active‑learning style acquisition: choose rows most likely to be
-- low performance using a simple decision tree model built on
-- previously evaluated rows.
-- ------------------------------------------------------------------
local function acquire_order(tbl, budget)
  local rows = tbl.rows          -- all candidate rows
  local lab = {}                 -- evaluated rows
  -- pick a random starting point
  local shuffled = keysort(rows, function(_) return rand() end)
  table.insert(lab, shuffled[1])

  -- model: leaf value from a tree built on labelled rows
  local function model(rowX, k)   -- k is number of goals (1)
    local t = Tree(tbl, lab)
    local val = t:leaf(tbl, rowX)
    return {val}
  end
  tbl.model = model

  local acquired = tbl:acquire(rows, budget, lab)

  -- Return the acquisition order (indices in original tbl.rows)
  local order = {}
  for _, r in ipairs(acquired) do
    for i, rr in ipairs(rows) do
      if r == rr then
        table.insert(order, {idx=i, row=r})
        break
      end
    end
  end
  return order
end

-- ------------------------------------------------------------------
-- Pretty‑print a row as a comma‑separated string
-- ------------------------------------------------------------------
local function row_to_string(row)
  local parts = {}
  for _, v in ipairs(row) do
    table.insert(parts, tostring(v))
  end
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- Main entry point
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end
  local budget = tonumber(arg[2]) or 10

  local tbl = load_tbl(path)

  -- ----------------- APACHE-13 -----------------
  print("== APACHE-13 ==")
  local best_i, best_val = find_best(tbl)
  if not best_i then
    print("  No performance data available.")
  else
    print(string.format("  Lowest performance value: %d", best_val))
    print(string.format("  Row %d: %s", best_i, row_to_string(tbl.rows[best_i])))
  end
  print()

  -- ----------------- APACHE-14 -----------------
  print("== APACHE-14 ==")
  print(string.format("  Evaluation budget: %d rows", budget))
  local order = acquire_order(tbl, budget)
  if #order == 0 then
    print("  No rows evaluated.")
  else
    print("  Suggested evaluation order (index, predicted performance):")
    for i, o in ipairs(order) do
      local row = o.row
      local pred = tbl:label(row)   -- force label for accuracy
      local val = row[tbl.cols.y[1].at]
      print(string.format("  %3d: idx=%3d  perf=%4s  pred=%4s",
                          i, o.idx, tostring(val), tostring(pred[tbl.cols.y[1].at])))
    end
  end
end

-- Run
main()
