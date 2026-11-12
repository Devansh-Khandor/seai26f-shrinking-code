#!/usr/bin/env lua
-- skill_07.lua
-- CAR-13 & CAR-14: Balanced vehicle selection & early inspection prioritisation
-- Usage: lua skill_07.lua <csv_path>
--
-- This script uses the ezr library bundled with the repository.
-- It prints the requested results for each CAR skill.
--
local ezr   = require("ezr")
local csv   = ezr.csv

--------------------------------------------------------------------
-- Helper utilities
--------------------------------------------------------------------
-- Return the column object for a given header name
local function col_of(tbl, name)
  for _,c in ipairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  return nil
end

-- Standardise a value against mean and sd (z‑score)
local function z_score(val, mu, sd)
  return (val - mu) / sd
end

-- Composite score: high mpg, low weight, high acceleration
local function score(row, idx_mpg, idx_lbs, idx_acc,
                     mu_mpg, sd_mpg, mu_lbs, sd_lbs, mu_acc, sd_acc)
  local mpg  = row[idx_mpg]
  local lbs  = row[idx_lbs]
  local acc  = row[idx_acc]
  if mpg == "?" or lbs == "?" or acc == "?" or
     mpg == nil or lbs == nil or acc == nil then
    return nil
  end
  local z_mpg = z_score(mpg, mu_mpg, sd_mpg)
  local z_lbs = -z_score(lbs, mu_lbs, sd_lbs)  -- lower better
  local z_acc = z_score(acc, mu_acc, sd_acc)
  return z_mpg + z_lbs + z_acc
end

--------------------------------------------------------------------
-- Main
--------------------------------------------------------------------
local function main(csv_path)
  -- Load table
  local tbl = ezr.Tbl(csv(csv_path))

  -- Column indices
  local col_mpg = col_of(tbl, "Mpg+")
  local col_lbs = col_of(tbl, "Lbs-")
  local col_acc = col_of(tbl, "Acc+")
  if not col_mpg or not col_lbs or not col_acc then
    error("Required columns missing")
  end
  local idx_mpg = col_mpg.at
  local idx_lbs = col_lbs.at
  local idx_acc = col_acc.at

  -- Statistics
  local mu_mpg = col_mpg:mid()
  local sd_mpg = col_mpg:div()
  local mu_lbs = col_lbs:mid()
  local sd_lbs = col_lbs:div()
  local mu_acc = col_acc:mid()
  local sd_acc = col_acc:div()

  -- Prepare data with row numbers (1‑based, excluding header)
  local rows = {}
  for i,r in ipairs(tbl.rows) do
    table.insert(rows, {idx=i, row=r})
  end

  ------------------------------------------------------------------
  -- CAR-13: Identify vehicle with strong balance, evaluating few candidates
  ------------------------------------------------------------------
  print("== CAR-13 ==")
  -- Random sample of ~20% (min 3)
  local sample_size = math.max(3, math.floor(#rows * 0.20))
  local shuffled = ezr.shuffle(rows)
  local sample = {}
  for i=1,sample_size do
    table.insert(sample, shuffled[i])
  end

  -- Evaluate sample
  local scored = {}
  for _,obj in ipairs(sample) do
    local s = score(obj.row, idx_mpg, idx_lbs, idx_acc,
                    mu_mpg, sd_mpg, mu_lbs, sd_lbs, mu_acc, sd_acc)
    if s then
      table.insert(scored, {idx=obj.idx, mpg=obj.row[idx_mpg],
                            lbs=obj.row[idx_lbs], acc=obj.row[idx_acc],
                            score=s})
    end
  end
  -- Sort by score descending
  table.sort(scored, function(a,b) return a.score > b.score end)

  -- Print top 3 in sample
  local display = math.min(3, #scored)
  for i=1,display do
    local r = scored[i]
    print(string.format(
      "  %d. Row %d: mpg=%.1f, weight=%.0f, accel=%.1f, score=%.3f",
      i, r.idx, r.mpg, r.lbs, r.acc, r.score))
  end
  print(string.format("Evaluated %d rows (sample) out of %d total rows.",
        sample_size, #rows))

  ------------------------------------------------------------------
  -- CAR-14: Prioritise inspection under limited budget
  ------------------------------------------------------------------
  print("\n== CAR-14 ==")
  -- Budget: 5 inspections
  local budget = 5
  local all_scored = {}
  for _,obj in ipairs(rows) do
    local s = score(obj.row, idx_mpg, idx_lbs, idx_acc,
                    mu_mpg, sd_mpg, mu_lbs, sd_lbs, mu_acc, sd_acc)
    if s then
      table.insert(all_scored, {idx=obj.idx, mpg=obj.row[idx_mpg],
                               lbs=obj.row[idx_lbs], acc=obj.row[idx_acc],
                               score=s})
    end
  end
  table.sort(all_scored, function(a,b) return a.score > b.score end)
  local to_show = math.min(budget, #all_scored)
  print(string.format("Top %d vehicles for early inspection (budget=%d):",
        to_show, budget))
  for i=1,to_show do
    local r = all_scored[i]
    print(string.format(
      "  %d. Row %d: mpg=%.1f, weight=%.0f, accel=%.1f, score=%.3f",
      i, r.idx, r.mpg, r.lbs, r.acc, r.score))
  end
end

--------------------------------------------------------------------
-- Entry point
--------------------------------------------------------------------
if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_07.lua <csv_path>\n")
end
