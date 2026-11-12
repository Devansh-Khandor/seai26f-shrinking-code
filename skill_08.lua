#!/usr/bin/env lua
-- Skill 08 – Apache configuration improvement recommendations
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_08.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Helper utilities
-- ------------------------------------------------------------------
local function get_col_by_name(name)
  for _, c in ipairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("Column '" .. name .. "' not found")
end

-- Number of differing independent columns between two rows
local function diff_count(r1, r2)
  local d = 0
  for _, c in ipairs(tbl.cols.x) do
    if r1[c.at] ~= r2[c.at] then d = d + 1 end
  end
  return d
end

-- Return a row string representation
local function row_to_str(r)
  local parts = {}
  for _, c in ipairs(tbl.cols.x) do
    parts[#parts+1] = c.name .. "=" .. tostring(r[c.at])
  end
  parts[#parts+1] = tbl.cols.klass.name .. "=" .. tostring(r[tbl.cols.klass.at])
  return table.concat(parts, ", ")
end

-- Find all rows with performance <= target
local function good_rows(target)
  local good = {}
  for _, r in ipairs(tbl.rows) do
    if r[tbl.cols.klass.at] <= target then table.insert(good, r) end
  end
  return good
end

-- ------------------------------------------------------------------
-- Gather overall statistics
-- ------------------------------------------------------------------
local perf_col = tbl.cols.klass
local perf_at  = perf_col.at
local perf_vals = {}
for _, r in ipairs(tbl.rows) do
  perf_vals[#perf_vals + 1] = r[perf_at]
end

local mean_perf = perf_col:mid()
local sd_perf   = perf_col:div()
local threshold = mean_perf - 2 * sd_perf  -- "good" performance threshold

-- ------------------------------------------------------------------
-- Identify a poorly performing configuration
-- ------------------------------------------------------------------
local poor_row = nil
local poor_perf = nil
for _, r in ipairs(tbl.rows) do
  local p = r[perf_at]
  if p > threshold then
    poor_row = r
    poor_perf = p
    break
  end
end

if not poor_row then
  print("== APACHE-15 ==")
  print("No configuration exceeds the good-performance threshold.")
  print("\n== APACHE-16 ==")
  print("No configuration exceeds the good-performance threshold.")
  os.exit(0)
end

-- ------------------------------------------------------------------
-- APACHE-15 – One‑setting change recommendation
-- ------------------------------------------------------------------
print("\n== APACHE-15 ==")

-- Search for a neighbor that differs in exactly one setting
-- and has lower performance.
local best_neighbor = nil
local best_perf = math.huge
local best_change = nil  -- {col, old, new}

for _, c in ipairs(tbl.cols.x) do
  for _, cand in ipairs(tbl.rows) do
    if cand ~= poor_row and diff_count(poor_row, cand) == 1 then
      local perf = cand[perf_at]
      if perf < poor_perf and perf < best_perf then
        best_perf = perf
        best_neighbor = cand
        -- find which column changed
        for _, col in ipairs(tbl.cols.x) do
          if poor_row[col.at] ~= cand[col.at] then
            best_change = {col=col.name, old=poor_row[col.at], new=cand[col.at]}
            break
          end
        end
      end
    end
  end
end

if best_neighbor then
  print(string.format("Recommend changing setting '%s' from %s to %s:",
          best_change.col, tostring(best_change.old), tostring(best_change.new)))
  print(string.format("Predicted performance after change: %s",
          ezr.round(best_perf)))
  print(string.format("Performance improvement: %s%%",
          ezr.round((poor_perf - best_perf) / poor_perf * 100)))
  print("Resulting configuration:")
  print(row_to_str(best_neighbor))
else
  -- Fallback: pick best row with lowest performance that differs in <= 2 settings
  local fallback_best = nil
  local fallback_perf = math.huge
  local fallback_dist = math.huge
  for _, cand in ipairs(tbl.rows) do
    if cand ~= poor_row then
      local d = diff_count(poor_row, cand)
      local perf = cand[perf_at]
      if perf < fallback_perf and (d == 1 or d == 2) then
        fallback_perf = perf
        fallback_best = cand
        fallback_dist = d
      end
    end
  end
  if fallback_best then
    print("No one‑setting change found with lower performance.")
    print(string.format("Best alternative differs in %d setting(s) with performance %s:",
            fallback_dist, ezr.round(fallback_perf)))
    print(row_to_str(fallback_best))
  else
    print("No better configuration found in the data set.")
  end
end

-- ------------------------------------------------------------------
-- APACHE-16 – Minimum‑change path to target performance
-- ------------------------------------------------------------------
print("\n== APACHE-16 ==")

-- Target: performance <= threshold
local good = good_rows(threshold)
if #good == 0 then
  print("No configuration meets the target performance threshold.")
else
  -- Find configuration with minimal differences to poor_row
  local best_goal = nil
  local best_goal_perf = math.huge
  local best_goal_dist = math.huge
  for _, cand in ipairs(good) do
    local d = diff_count(poor_row, cand)
    local perf = cand[perf_at]
    if d < best_goal_dist or (d == best_goal_dist and perf < best_goal_perf) then
      best_goal_dist = d
      best_goal_perf = perf
      best_goal = cand
    end
  end

  print(string.format("Target performance <= %s achieved with %d setting change(s).",
          ezr.round(threshold), best_goal_dist))
  print(string.format("Recommended configuration:")
          .. row_to_str(best_goal))
  print(string.format("Performance: %s (improvement %s%%)",
          ezr.round(best_goal_perf),
          ezr.round((poor_perf - best_goal_perf) / poor_perf * 100)))
  -- List changed settings
  if best_goal_dist > 0 then
    print("Changed settings:")
    for _, c in ipairs(tbl.cols.x) do
      if poor_row[c.at] ~= best_goal[c.at] then
        print(string.format(" - %s: %s -> %s",
                c.name,
                tostring(poor_row[c.at]),
                tostring(best_goal[c.at])))
      end
    end
  else
    print("No changes required: poor configuration already meets target.")
  end
end
