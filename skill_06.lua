#!/usr/bin/env lua
-- Skill 06: Explain effort estimates and risk differences
-- Usage: lua skill_06.lua <csv_path>

local ezr = require "ezr"
local show = ezr.show
local push = ezr.push
local sorted = ezr.sorted
local keys = ezr.keys
local rand = ezr.rand
local map = ezr.map
local slice = ezr.slice

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- ------------------------------------------------------------------
-- Helper: return the first column in tbl.cols.all with this name
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

-- Helper: compute statistics for numeric column
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
      -- ignore rows with missing x data
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
-- Find all proposals (rows with missing effort)
local function find_proposals(tbl)
  local proposals = {}
  local effort_col = find_col(tbl, "Effort-")
  for _,r in ipairs(tbl.rows) do
    if r[effort_col.at] == "?" then
      push(proposals, r)
    end
  end
  return proposals
end

-- ------------------------------------------------------------------
-- Compute top differences between two proposals
local function diff_between(tbl, p1, p2)
  local diffs = {}
  for _,c in ipairs(tbl.cols.x) do
    local v1 = p1[c.at]
    local v2 = p2[c.at]
    if type(v1) == "number" and type(v2) == "number" then
      local min = c.n > 0 and c.n and c.n or 1 -- fallback
      local range = c:div() * 2 -- approximate range via std dev
      if range == 0 then range = 1 end
      local norm = math.abs(v1 - v2) / range
      push(diffs, {col=c, kind="num", val1=v1, val2=v2, norm=norm})
    else
      local diff = v1 ~= v2
      push(diffs, {col=c, kind="sym", val1=v1, val2=v2, diff=diff})
    end
  end
  table.sort(diffs, function(a,b)
    if a.kind == b.kind then
      if a.kind=="num" then return a.norm > b.norm end
      return a.diff > b.diff
    else
      return a.kind == "num"
    end
  end)
  return diffs
end

-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_06.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  -- --------------------------------------------------------------
  -- NASA‑11 : explain effort estimate
  print("== NASA-11 ==")
  local effort_col = find_col(tbl, "Effort-")
  local effort_stats = col_stats(tbl, effort_col)
  print(string.format("Historical Effort- mean=%.1f sd=%.1f",
        effort_stats.mean, effort_stats.sd))

  local proposals = find_proposals(tbl)
  if #proposals == 0 then
    print("No proposals with missing Effort- to analyse.")
  else
    local prop = proposals[1]
    local pred = knn_predict(tbl, prop, effort_col, 3)
    print(string.format("Proposal id %s estimated Effort-: %.1f",
          prop[1], pred or effort_stats.mean))

    -- find top 3 similar projects
    local all_neigh = {}
    for _,r in ipairs(tbl.rows) do
      if r[effort_col.at] ~= "?" and r ~= prop then
        local d = tbl:distx(prop, r)
        push(all_neigh, {row=r, dist=d})
      end
    end
    sorted(all_neigh, function(a,b) return a.dist < b.dist end)
    local neigh = slice(all_neigh, 1, 3)
    print("Top 3 similar projects:")
    for i,n in ipairs(neigh) do
      print(string.format("  %d. id %s, Effort- %.1f, dist %.3f",
            i, n.row[1], n.row[effort_col.at], n.dist))
    end
  end

  -- --------------------------------------------------------------
  -- NASA‑12 : compare two proposals
  print("\n== NASA-12 ==")
  if #proposals < 2 then
    print("Not enough proposals to compare.")
  else
    local p1, p2 = proposals[1], proposals[2]
    print(string.format("Comparing proposals %s and %s",
          p1[1], p2[1]))

    local diffs = diff_between(tbl, p1, p2)
    print("Key differences (most significant first):")
    local limit = math.min(5, #diffs)
    for i=1,limit do
      local d = diffs[i]
      if d.kind=="num" then
        print(string.format("  %d. %s: %.1f vs %.1f (Δ=%.2f)",
              i, d.col.name, d.val1, d.val2, d.norm))
      else
        if d.diff then
          print(string.format("  %d. %s: %s vs %s",
                i, d.col.name, d.val1, d.val2))
        end
      end
    end

    -- simple trade‑off summary
    print("\nTrade‑off summary:")
    local size_col = find_col(tbl, "Kloc+")
    local def_col  = find_col(tbl, "Defects-")
    print(string.format("  Size: %s (%d) vs %s (%d)",
          p1[1], p1[size_col.at], p2[1], p2[size_col.at]))
    print(string.format("  Defects: %s (%.1f) vs %s (%.1f)",
          p1[1], p1[def_col.at], p2[1], p2[def_col.at]))
    print("  Generally, a larger project tends to have higher effort but may also have fewer defects if well managed.")
  end
end

main()
