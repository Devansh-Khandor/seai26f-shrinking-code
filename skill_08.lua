#!/usr/bin/env lua
-- skill_08.lua
-- Run: lua skill_08.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-15 – For a project proposal predicted to require excessive effort,
--             identify plausible changes to adjustable planning or process
--             characteristics that might reduce its predicted effort.
--   NASA-16 – For a project proposal with a high predicted defect count,
--             identify the smallest plausible combination of changes to
--             adjustable planning or process characteristics that could
--             improve its predicted quality, while explaining potential
--             tradeoffs.

local ezr = require "ezr"

-- Helpers from ezr
local csv       = ezr.csv
local Tbl       = ezr.Tbl
local push      = ezr.push
local sorted    = ezr.sorted
local sqrt      = math.sqrt

-- ------------------------------------------------------------------
-- Utility helpers
-- ------------------------------------------------------------------
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  return {
    min  = col.min,
    max  = col.max,
    mean = col:mid(),
    std  = col:div()
  }
end

local function col_mean(rows, at)
  local sum, n = 0, 0
  for _, r in ipairs(rows) do
    local v = r[at]
    if v ~= "?" then
      sum = sum + v
      n = n + 1
    end
  end
  return n > 0 and sum / n or nil
end

local function sym_mode(rows, at)
  local freq = {}
  local maxc, mode = 0, nil
  for _, r in ipairs(rows) do
    local v = r[at]
    if v ~= "?" then
      freq[v] = (freq[v] or 0) + 1
      if freq[v] > maxc then
        maxc, mode = freq[v], v
      end
    end
  end
  return mode, maxc
end

-- Analyze a column comparing high-group vs overall
local function analyze_column(tbl, high_rows, overall_rows, at)
  local col = tbl.cols.all[at]
  local mt = getmetatable(col).__index
  if mt == ezr.NUM then
    local overall_mean = col_mean(overall_rows, at)
    local high_mean    = col_mean(high_rows, at)
    if high_mean and overall_mean then
      return {
        name    = col.name,
        type    = "num",
        high    = high_mean,
        overall = overall_mean,
        diff    = high_mean - overall_mean,
        pct     = (high_mean - overall_mean) / overall_mean * 100
      }
    end
  elseif mt == ezr.SYM then
    local overall_mode, overall_cnt = sym_mode(overall_rows, at)
    local high_mode, high_cnt = sym_mode(high_rows, at)
    if high_mode and overall_mode then
      return {
        name        = col.name,
        type        = "sym",
        high_mode   = high_mode,
        high_cnt    = high_cnt,
        overall_mode= overall_mode,
        overall_cnt = overall_cnt,
        diff_cnt    = high_cnt - overall_cnt
      }
    end
  end
end

-- ------------------------------------------------------------------
-- Main entry point
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_08.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))
local n = #tbl.rows

-- ------------------------------------------------------------------
-- NASA-15: Excessive effort
-- ------------------------------------------------------------------
local effort_at   = 27
local effort_stats = col_stats(tbl, effort_at)
local eff_se      = effort_stats.std / sqrt(n)
local eff_thresh  = effort_stats.mean + 1.96 * eff_se

local high_eff_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[effort_at] > eff_thresh then
    push(high_eff_rows, r)
  end
end

local high_eff_count = #high_eff_rows

local analyses = {}
for at, _ in pairs(tbl.cols.all) do
  if at ~= 1 and at ~= 2 and at ~= 3 and at ~= 26 and at ~= 27 and at ~= 28 and at ~= 29 then
    local a = analyze_column(tbl, high_eff_rows, tbl.rows, at)
    if a then push(analyses, a) end
  end
end

-- Keep only characteristics where high group tends to be worse
local suggestions_eff = {}
for _, a in ipairs(analyses) do
  if a.type == "num" and a.diff > 0 then
    push(suggestions_eff, a)
  elseif a.type == "sym" and a.diff_cnt ~= 0 then
    push(suggestions_eff, a)
  end
end

-- Sort by most positive numeric diff (or symbolic mode change magnitude)
suggestions_eff = sorted(suggestions_eff, function(a,b)
  if a.type == "num" and b.type == "num" then
    return a.diff > b.diff
  elseif a.type == "sym" and b.type == "sym" then
    return math.abs(a.diff_cnt) > math.abs(b.diff_cnt)
  else
    return a.type == "num"
  end
end)

print("== NASA-15 ==")
print(string.format(
  "High‑effort threshold: %.2f (mean=%.2f, SE=%.2f)",
  eff_thresh, effort_stats.mean, eff_se))
print(string.format("Projects exceeding threshold: %d out of %d", high_eff_count, n))
print("Suggested adjustments (top 3):")
for i=1,math.min(3,#suggestions_eff) do
  local s = suggestions_eff[i]
  if s.type=="num" then
    print(string.format(
      "%d) Reduce %s: current mean=%.2f, overall mean=%.2f, Δ=%.2f (%.1f%% reduction)",
      i, s.name, s.high, s.overall, s.diff, s.pct))
  else
    local change = s.diff_cnt>0 and "shift to" or "change from"
    print(string.format(
      "%d) %s %s: current mode=%s (%d), overall mode=%s (%d), Δ=%d",
      i, change, s.name, s.high_mode, s.high_cnt,
      s.overall_mode, s.overall_cnt, s.diff_cnt))
  end
end
if #suggestions_eff==0 then print("  No clear adjustment found.") end

-- ------------------------------------------------------------------
-- NASA-16: High predicted defect count
-- ------------------------------------------------------------------
local defect_at   = 28
local defect_stats = col_stats(tbl, defect_at)
local def_se      = defect_stats.std / sqrt(n)
local def_thresh  = defect_stats.mean + 1.96 * def_se

local high_def_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[defect_at] > def_thresh then
    push(high_def_rows, r)
  end
end

local high_def_count = #high_def_rows

local analyses_def = {}
for at, _ in pairs(tbl.cols.all) do
  if at ~= 1 and at ~= 2 and at ~= 3 and at ~= 26 and at ~= 27 and at ~= 28 and at ~= 29 then
    local a = analyze_column(tbl, high_def_rows, tbl.rows, at)
    if a then push(analyses_def, a) end
  end
end

-- Filter for characteristics that worsen defects
local suggestions_def = {}
for _, a in ipairs(analyses_def) do
  if a.type=="num" and a.diff>0 then
    push(suggestions_def, a)
  elseif a.type=="sym" and a.diff_cnt ~= 0 then
    push(suggestions_def, a)
  end
end

-- Sort by impact
suggestions_def = sorted(suggestions_def, function(a,b)
  if a.type=="num" and b.type=="num" then
    return a.diff > b.diff
  elseif a.type=="sym" and b.type=="sym" then
    return math.abs(a.diff_cnt) > math.abs(b.diff_cnt)
  else
    return a.type=="num"
  end
end)

print("\n== NASA-16 ==")
print(string.format(
  "High‑defect threshold: %.2f (mean=%.2f, SE=%.2f)",
  def_thresh, defect_stats.mean, def_se))
print(string.format("Projects exceeding threshold: %d out of %d", high_def_count, n))
print("Suggested changes (top 3):")
for i=1,math.min(3,#suggestions_def) do
  local s = suggestions_def[i]
  if s.type=="num" then
    print(string.format(
      "%d) Reduce %s: current mean=%.2f, overall mean=%.2f, Δ=%.2f (%.1f%% reduction)",
      i, s.name, s.high, s.overall, s.diff, s.pct))
  else
    local change = s.diff_cnt>0 and "shift to" or "change from"
    print(string.format(
      "%d) %s %s: current mode=%s (%d), overall mode=%s (%d), Δ=%d",
      i, change, s.name, s.high_mode, s.high_cnt,
      s.overall_mode, s.overall_cnt, s.diff_cnt))
  end
end
if #suggestions_def==0 then print("  No clear change found.") end

-- ------------------------------------------------------------------
-- Smallest plausible combination (example)
-- ------------------------------------------------------------------
local combo = {}
-- pick first numeric and first symbolic if available
local numeric = nil
local symbolic = nil
for _,s in ipairs(suggestions_def) do
  if not numeric and s.type=="num" then numeric=s end
  if not symbolic and s.type=="sym" then symbolic=s end
end
if numeric or symbolic then
  print("\nExample minimal combination to improve quality:")
  if numeric then
    print(string.format(
      "  • Reduce %s from %.2f to %.2f",
      numeric.name, numeric.high, numeric.overall))
  end
  if symbolic then
    print(string.format(
      "  • Shift %s from %s to %s",
      symbolic.name, symbolic.high_mode, symbolic.overall_mode))
  end
  print("Potential tradeoffs: changes may increase schedule pressure or cost, but reduce defect risk.")
else
  print("\nNo actionable combination identified.")
end
