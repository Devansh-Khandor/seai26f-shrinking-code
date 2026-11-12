#!/usr/bin/env lua
-- Skill 05: Analyze poor configurations
-- APACHE-09 : which single settings are most associated with bad performance
-- APACHE-10 : which combinations of enabled settings appear most often in poor rows

local ezr = require "ezr"

local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Compute overall performance statistics (mean, sd, etc.)
-- ------------------------------------------------------------------
local function performance_stats(tbl)
  local perf_at = tbl.cols.y[1].at
  local mids = tbl:mids()
  local mean   = mids[perf_at]
  local sd     = tbl.cols.all[perf_at]:div()

  local sorted_rows = ezr.sorted(tbl.rows,
    function(r) return r[perf_at] end)

  local n = #sorted_rows
  local min_perf = sorted_rows[1][perf_at]
  local max_perf = sorted_rows[n][perf_at]
  local median_perf = sorted_rows[math.floor(n/2)+1][perf_at]

  return {
    mean   = mean,
    median = median_perf,
    sd     = sd,
    best   = min_perf,
    worst  = max_perf,
    sorted = sorted_rows,
    at     = perf_at,
    mids   = mids
  }
end

-- ------------------------------------------------------------------
-- Return rows that are poorly performing (above mean+sd)
-- ------------------------------------------------------------------
local function poor_rows(tbl)
  local stats = performance_stats(tbl)
  local threshold = stats.mean + stats.sd   -- 1‑sigma above mean
  local poor = {}
  local perf_at = tbl.cols.y[1].at
  for _, r in ipairs(tbl.rows) do
    if r[perf_at] > threshold then
      table.insert(poor, r)
    end
  end
  return poor
end

-- ------------------------------------------------------------------
-- APACHE-09: association of individual settings with bad performance
-- ------------------------------------------------------------------
local function setting_associations(tbl)
  local perf_at = tbl.cols.y[1].at
  local assoc = {}
  for _, c in ipairs(tbl.cols.x) do
    local en, dis = {}, {}
    for _, r in ipairs(tbl.rows) do
      local p = r[perf_at]
      if r[c.at] == 1 then
        en[#en+1] = p
      elseif r[c.at] == 0 then
        dis[#dis+1] = p
      end
    end
    local mean_en = #en>0 and ezr.sum(en, function(v) return v end)/#en or nil
    local mean_dis = #dis>0 and ezr.sum(dis, function(v) return v end)/#dis or nil
    local diff = (mean_en and mean_dis) and mean_dis - mean_en or nil
    assoc[c.name] = {enabled=mean_en, disabled=mean_dis, diff=diff, n_en=#en, n_dis=#dis}
  end
  return assoc
end

-- ------------------------------------------------------------------
-- APACHE-10: frequent combinations among poorly performing rows
-- ------------------------------------------------------------------
local function pair_frequent_in_poor(tbl, poor)
  local n_total = #tbl.rows
  local n_poor = #poor
  local pairs = {}
  for i=1,#tbl.cols.x-1 do
    local c1 = tbl.cols.x[i]
    for j=i+1,#tbl.cols.x do
      local c2 = tbl.cols.x[j]
      local count_all, count_poor = 0, 0
      for _, r in ipairs(tbl.rows) do
        if r[c1.at]==1 and r[c2.at]==1 then
          count_all = count_all + 1
        end
      end
      for _, r in ipairs(poor) do
        if r[c1.at]==1 and r[c2.at]==1 then
          count_poor = count_poor + 1
        end
      end
      if count_poor > 0 then
        local prop_all = count_all / n_total
        local prop_poor = count_poor / n_poor
        local lift = prop_poor / (prop_all + 1e-12)
        pairs[#pairs+1] = {
          c1=c1.name, c2=c2.name,
          n_poor=count_poor, p_poor=prop_poor,
          n_all=count_all, p_all=prop_all,
          lift=lift
        }
      end
    end
  end
  return pairs
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_05.lua <csv path>")

  -- Load the data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- ----------------------------------------------------------------
  -- APACHE-09  – single setting associations
  -- ----------------------------------------------------------------
  print("== APACHE-09 ==")
  local assoc = setting_associations(tbl)

  -- Sort settings by absolute diff descending (worst first)
  local sorted = {}
  for name, v in pairs(assoc) do
    if v.diff then
      sorted[#sorted+1] = {name=name, en=v.enabled, dis=v.disabled, diff=v.diff}
    end
  end
  table.sort(sorted, function(a,b) return a.diff > b.diff end)

  -- Show top 5
  print(string.format("%-20s %-10s %-10s %-10s", "Setting", "Enabled", "Disabled", "Diff"))
  for i=1, math.min(5, #sorted) do
    local s = sorted[i]
    print(string.format("%-20s %-10s %-10s %-10s",
      s.name,
      pretty(s.en),
      pretty(s.dis),
      pretty(s.diff)))
  end

  -- ----------------------------------------------------------------
  -- APACHE-10  – frequent combinations in poor rows
  -- ----------------------------------------------------------------
  print("\n== APACHE-10 ==")
  local poor = poor_rows(tbl)
  local pairs = pair_frequent_in_poor(tbl, poor)

  -- Sort by lift (how much more frequent in poor set)
  table.sort(pairs, function(a,b) return a.lift > b.lift end)

  print(string.format("%-30s %-6s %-6s %-6s %-6s %-6s %-6s",
    "Pair (enabled)", "Npoor", "Ppoor", "Nall", "Pall", "Lift"))
  for i=1, math.min(5, #pairs) do
    local p = pairs[i]
    print(string.format("%-30s %-6d %-6.3f %-6d %-6.3f %-6.3f",
      p.c1 .. ", " .. p.c2,
      p.n_poor, p.p_poor,
      p.n_all, p.p_all,
      p.lift))
  end
end

-- Execute when run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_05.lua$") then
  main()
end
