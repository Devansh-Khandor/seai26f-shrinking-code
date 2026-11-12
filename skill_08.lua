#!/usr/bin/env lua
-- skill_08.lua
-- CAR-15 & CAR-16: Alternative vehicle selection & improvement analysis
-- Requires only the bundled ezr library.
--
-- Usage:  lua skill_08.lua <path-to-auto93.csv>
--
-- The script prints, for each CAR skill, a header followed by at least three
-- lines of results, including the actual numbers.

local ezr   = require("ezr")
local csv   = ezr.csv
local math  = math

-- --------------------------------------------------------------------
-- Helpers
-- --------------------------------------------------------------------

-- Return a list of numeric columns (excluding the target columns)
local function numeric_features(tbl, skip_names)
  skip_names = skip_names or {}
  local cols = {}
  for _, c in ipairs(tbl.cols.all) do
    if (c.__index == NUM or c.__index == ezr.NUM) then
      if not skip_names[c.name] then
        table.insert(cols, c)
      end
    end
  end
  return cols
end

-- Compute the desirability (mpg / lbs) of a row
local function desirability(row, mpg_idx, lbs_idx)
  local mpg = row[mpg_idx]
  local lbs = row[lbs_idx]
  if mpg == "?" or lbs == "?" or not mpg or not lbs then return nil end
  return mpg / lbs
end

-- Euclidean distance over numeric features
local function distance(row1, row2, feats)
  local sum = 0
  for _,c in ipairs(feats) do
    local v1, v2 = row1[c.at], row2[c.at]
    if v1 == "?" or v2 == "?" or not v1 or not v2 then
      return nil
    end
    local d = v1 - v2
    sum = sum + d * d
  end
  return math.sqrt(sum)
end

-- Find the row with the lowest desirability
local function find_worst_row(tbl, mpg_idx, lbs_idx)
  local worst = nil
  local worst_val = 1e18
  for _, r in ipairs(tbl.rows) do
    local d = desirability(r, mpg_idx, lbs_idx)
    if d and d < worst_val then
      worst_val = d
      worst = r
    end
  end
  return worst, worst_val
end

-- Find the nearest neighbours to a given row, sorted by distance
local function find_neighbours(target, tbl, feats, count)
  local neighs = {}
  for _, r in ipairs(tbl.rows) do
    if r ~= target then
      local d = distance(target, r, feats)
      if d then
        table.insert(neighs, {row=r, dist=d})
      end
    end
  end
  table.sort(neighs, function(a,b) return a.dist < b.dist end)
  return neighs
end

-- Find a row that fails at least one performance target
local function find_non_targeted_row(tbl, mpg_col, lbs_col)
  local median_mpg = ezr.median(ezr.numeric_values(tbl, mpg_col))
  local median_lbs = ezr.median(ezr.numeric_values(tbl, lbs_col))
  for i, r in ipairs(tbl.rows) do
    local mpg = r[mpg_col.at]
    local lbs = r[lbs_col.at]
    if mpg == "?" or lbs == "?" or not mpg or not lbs then
      return nil
    end
    if mpg < median_mpg or lbs > median_lbs then
      return r
    end
  end
  return nil
end

-- Find the best alternative for a target‑violating row
local function find_best_alternative(vrow, tbl, mpg_col, lbs_col, feats)
  local median_mpg = ezr.median(ezr.numeric_values(tbl, mpg_col))
  local median_lbs = ezr.median(ezr.numeric_values(tbl, lbs_col))
  local best = nil
  local best_score = -1
  local best_dist  = 1e18

  for _, r in ipairs(tbl.rows) do
    if r ~= vrow then
      local mpg = r[mpg_col.at]
      local lbs = r[lbs_col.at]
      if mpg == "?" or lbs == "?" or not mpg or not lbs then
        goto cont
      end
      local targets = 0
      if mpg >= median_mpg then targets = targets + 1 end
      if lbs <= median_lbs then targets = targets + 1 end
      local d = distance(vrow, r, feats)
      if not d then goto cont end

      if targets > best_score or
         (targets == best_score and d < best_dist) then
        best = r
        best_score = targets
        best_dist = d
      end
    end
    ::cont::
  end
  return best, best_score, best_dist
end

-- Explain differences between two rows
local function explain_differences(a, b, feats)
  local parts = {}
  for _,c in ipairs(feats) do
    local va, vb = a[c.at], b[c.at]
    if va ~= "?" and vb ~= "?" and va and vb then
      local diff = vb - va
      table.insert(parts,
        string.format("%s: %.2f → %.2f (Δ=%.2f)",
          c.name, va, vb, diff))
    end
  end
  return table.concat(parts, "; ")
end

-- --------------------------------------------------------------------
-- Main
-- --------------------------------------------------------------------

local function main(csv_path)
  local tbl = ezr.Tbl(csv(csv_path))

  -- Find relevant column indices
  local mpg_col, lbs_col, acc_col, vol_col, hp_col
  for _,c in ipairs(tbl.cols.all) do
    if c.name == "Mpg+" then mpg_col = c end
    if c.name == "Lbs-" then lbs_col = c end
    if c.name == "Acc+" then acc_col = c end
    if c.name == "Volume" then vol_col = c end
    if c.name == "HpX" then hp_col = c end
  end
  if not mpg_col or not lbs_col then
    error("Missing required columns")
  end

  local feats = numeric_features(tbl, {
    ["Mpg+"] = true, ["Lbs-"] = true
  })

  -- ------------------------------------------------------------------
  -- CAR-15: Poorly rated vehicle + better alternatives
  -- ------------------------------------------------------------------
  print("== CAR-15 ==")
  local worst, worst_val = find_worst_row(tbl, mpg_col.at, lbs_col.at)
  if not worst then
    print("No valid rows found.")
  else
    local neighs = find_neighbours(worst, tbl, feats, 10)
    local printed = 0
    for _,n in ipairs(neighs) do
      local d_val = desirability(n.row, mpg_col.at, lbs_col.at)
      if d_val and d_val > worst_val then
        printed = printed + 1
        print(string.format(
          "%d. Row %d: mpg=%.1f, lbs=%.0f, desir=%.4f",
          printed,
          tbl.rows[1] == worst and 1 or table.find(tbl.rows, worst),
          worst[mpg_col.at], worst[lbs_col.at], worst_val))
        print(string.format(
          "   Alternative Row %d: mpg=%.1f, lbs=%.0f, desir=%.4f",
          tbl.rows[1] == n.row and 1 or table.find(tbl.rows, n.row),
          n.row[mpg_col.at], n.row[lbs_col.at], d_val))
        print("   Feature differences: " ..
          explain_differences(worst, n.row, feats))
        if printed >= 3 then break end
      end
    end
    if printed < 3 then
      print("  Less than 3 better alternatives available.")
    end
  end

  -- ------------------------------------------------------------------
  -- CAR-16: Vehicle not meeting targets + closest better alternative
  -- ------------------------------------------------------------------
  print("\n== CAR-16 ==")
  local violator = find_non_targeted_row(tbl, mpg_col, lbs_col)
  if not violator then
    print("All vehicles meet performance targets.")
  else
    local alt, score, dist = find_best_alternative(violator, tbl,
                                                   mpg_col, lbs_col, feats)
    if not alt then
      print("No suitable alternative found.")
    else
      local vrow_id = table.find(tbl.rows, violator)
      local alt_id = table.find(tbl.rows, alt)
      local v_mpg, v_lbs = violator[mpg_col.at], violator[lbs_col.at]
      local a_mpg, a_lbs = alt[mpg_col.at], alt[lbs_col.at]
      local median_mpg = ezr.median(ezr.numeric_values(tbl, mpg_col))
      local median_lbs = ezr.median(ezr.numeric_values(tbl, lbs_col))

      print(string.format(
        "Vehicle %d: mpg=%.1f, lbs=%.0f  (targets: mpg>=%.1f, lbs<=%.0f)",
        vrow_id, v_mpg, v_lbs, median_mpg, median_lbs))
      print(string.format(
        "Closest better alternative %d: mpg=%.1f, lbs=%.0f  (targets met: %d)",
        alt_id, a_mpg, a_lbs, score))
      print("Feature differences: " ..
        explain_differences(violator, alt, feats))
    end
  end
end

-- --------------------------------------------------------------------
-- Entry point
-- --------------------------------------------------------------------
if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_08.lua <csv_path>\n")
end
