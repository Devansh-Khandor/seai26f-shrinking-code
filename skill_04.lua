#!/usr/bin/env lua
-- Skill 04: New proposal assessment and dynamic history update
-- Usage: lua skill_04.lua <csv_path> [<csv_new_completed_path>]

local ezr = require "ezr"
local show  = ezr.show
local push  = ezr.push
local sorted = ezr.sorted
local map   = ezr.map
local rand  = ezr.rand

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- ------------------------------------------------------------------
-- Helper functions (copy of what was in skill_01 / 02)

local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

local function col_stats(tbl, col)
  local vals = map(tbl.rows, function(r) return r[col.at] end)
  local sorted_vals = sorted(vals)
  local n = #sorted_vals
  local mean = col.mu
  local sd   = col.div()
  local med  = sorted_vals[(n + 1) // 2]
  return {
    name   = col.name,
    mean   = mean,
    median = med,
    sd     = sd,
    min    = sorted_vals[1],
    max    = sorted_vals[n]
  }
end

-- ------------------------------------------------------------------
-- k‑NN prediction for a single goal
local function knn_predict(tbl, proposal, goal_col, k)
  local labeled = {}
  for _,r in ipairs(tbl.rows) do
    if r[goal_col.at] ~= "?" then
      -- ignore rows with missing x data (unlikely for NASA‑93)
      local missing = false
      for _,c in ipairs(tbl.cols.x) do
        if r[c.at] == "?" then missing = true; break end
      end
      if not missing then push(labeled, r) end
    end
  end
  if #labeled == 0 then return nil end

  local function dist_to(r) return tbl:distx(proposal, r) end

  local sorted_rows = sorted(labeled, function(a,b)
    return dist_to(a) < dist_to(b)
  end)

  local k_use = math.min(k, #sorted_rows)
  local w_sum, val_sum = 0, 0
  for i=1,k_use do
    local r = sorted_rows[i]
    local d = dist_to(r)
    local w = 1/(d+1e-6)
    w_sum   = w_sum + w
    val_sum = val_sum + w * r[goal_col.at]
  end
  return val_sum / w_sum
end

-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_04.lua <csv_path> [<csv_new_completed_path>]\n")
    os.exit(1)
  end

  -- Load base table
  local tbl = ezr.Tbl(ezr.csv(path))

  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  local effort_stats = col_stats(tbl, effort_col)
  local defect_stats = col_stats(tbl, defect_col)

  --------------------------------------------------------------
  -- NASA‑07 : assess new proposals
  print("== NASA-07 ==")
  print(string.format("Historical Effort- mean=%.1f sd=%.1f",
        effort_stats.mean, effort_stats.sd))
  print(string.format("Historical Defects- mean=%.1f sd=%.1f",
        defect_stats.mean, defect_stats.sd))

  -- find proposals (missing a goal)
  local proposals = {}
  for _, r in ipairs(tbl.rows) do
    if r[effort_col.at] == "?" or r[defect_col.at] == "?" then
      push(proposals, r)
    end
  end
  print(string.format("Proposals found: %d", #proposals))

  for _, r in ipairs(proposals) do
    local id = r[1]
    local pred_eff = knn_predict(tbl, r, effort_col, 5)
    local pred_def = knn_predict(tbl, r, defect_col, 5)
    local eff_high = pred_eff and pred_eff > effort_stats.mean + 2*effort_stats.sd
    local def_high = pred_def and pred_def > defect_stats.mean + 2*defect_stats.sd
    print(string.format(
      "Proposal id %s: Effort=%.1f%s  Defects=%.1f%s",
      id,
      pred_eff or 0, eff_high and " (HIGH)" or "",
      pred_def or 0, def_high and " (HIGH)" or ""
    ))
  end

  --------------------------------------------------------------
  -- NASA‑08 : incorporate additional completed projects
  local new_path = arg[2]
  if new_path then
    local new_tbl = ezr.Tbl(ezr.csv(new_path))
    for _,r in ipairs(new_tbl.rows) do
      tbl:add(r)
    end
    -- recompute statistics
    effort_stats = col_stats(tbl, effort_col)
    defect_stats = col_stats(tbl, defect_col)
    print("\n== NASA-08 ==")
    print(string.format("Updated Effort- mean=%.1f sd=%.1f",
          effort_stats.mean, effort_stats.sd))
    print(string.format("Updated Defects- mean=%.1f sd=%.1f",
          defect_stats.mean, defect_stats.sd))
  end
end

main()
