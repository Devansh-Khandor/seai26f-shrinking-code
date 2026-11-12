#!/usr/bin/env lua
-- skill_05.lua
-- CAR-09 & CAR-10: Association analysis of poor fuel‑efficiency
-- and overall performance characteristics.
--
-- Usage:  lua skill_05.lua <path-to-auto93.csv>
--
-- The script prints a header for each skill followed by
-- at least three lines of numeric results, using only the
-- ezr library bundled in this repository.

local ezr   = require("ezr")
local csv   = ezr.csv

-- ---------- utilities ------------------------------------------------

-- Collect numeric values for a column, ignoring "?" and nil
local function numeric_values(tbl, col)
  local out = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[col.at]
    if v ~= "?" and v ~= nil then
      out[#out + 1] = v
    end
  end
  return out
end

-- Return median of a numeric list
local function median(vals)
  local n = #vals
  if n == 0 then return nil end
  local sorted = ezr.sorted(vals)
  if n % 2 == 1 then
    return sorted[(n + 1) // 2]
  else
    return (sorted[n // 2] + sorted[n // 2 + 1]) / 2
  end
end

-- Return standard deviation of a numeric list
local function stddev(vals, mu)
  local n = #vals
  if n <= 1 then return 0 end
  mu = mu or (function()
    local s = 0
    for _, v in ipairs(vals) do s = s + v end
    return s / n
  end)()
  local ss = 0
  for _, v in ipairs(vals) do ss = ss + (v - mu) ^ 2 end
  return math.sqrt(ss / (n - 1))
end

-- Pearson correlation between two numeric lists
local function correlation(x, y)
  local nx = #x
  if nx ~= #y or nx == 0 then return nil end
  local mx, my = 0, 0
  for i = 1, nx do
    mx = mx + x[i]
    my = my + y[i]
  end
  mx = mx / nx
  my = my / nx
  local sx, sy, cov = 0, 0, 0
  for i = 1, nx do
    local dx = x[i] - mx
    local dy = y[i] - my
    sx = sx + dx * dx
    sy = sy + dy * dy
    cov = cov + dx * dy
  end
  local denom = math.sqrt(sx * sy)
  return denom == 0 and 0 or cov / denom
end

-- ------------------------------------------------------------------
-- 1. CAR-09 : Attributes most strongly linked to low MPG
-- ------------------------------------------------------------------

local function car09(tbl)
  print("== CAR-09 ==")

  -- Find MPG column
  local mpg_col
  for _, col in ipairs(tbl.cols.all) do
    if col.name == "Mpg+" then mpg_col = col end
  end
  if not mpg_col then error("Missing Mpg+ column") end

  -- Gather all MPG values
  local mpg_vals = numeric_values(tbl, mpg_col)
  local mpg_med = median(mpg_vals)
  local mpg_sd  = stddev(mpg_vals, mpg_med)

  -- Define low‑MPG group: < median - sd
  local low_threshold = mpg_med - mpg_sd
  local low_rows = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[mpg_col.at]
    if v ~= "?" and v ~= nil and v < low_threshold then
      low_rows[#low_rows + 1] = r
    end
  end
  if #low_rows == 0 then
    low_threshold = mpg_med - (mpg_sd / 2)
    for _, r in ipairs(tbl.rows) do
      local v = r[mpg_col.at]
      if v ~= "?" and v ~= nil and v < low_threshold then
        low_rows[#low_rows + 1] = r
      end
    end
  end
  if #low_rows == 0 then
    -- fallback to worst 10%
    local n = math.floor(0.1 * #tbl.rows)
    low_rows = {}
    table.sort(tbl.rows, function(a, b) return a[mpg_col.at] < b[mpg_col.at] end)
    for i = 1, n do low_rows[#low_rows + 1] = tbl.rows[i] end
  end

  -- Compute correlation of each numeric column with MPG
  local numeric_cols = {}
  for _, col in ipairs(tbl.cols.all) do
    if col.name ~= "Mpg+" and (col.__index == NUM or col.__index == ezr.NUM) then
      numeric_cols[#numeric_cols + 1] = col
    end
  end

  local corrs = {}
  for _, col in ipairs(numeric_cols) do
    local vals = numeric_values(tbl, col)
    local mpv  = numeric_values(tbl, mpg_col)
    local r = correlation(vals, mpv)
    corrs[#corrs + 1] = {name = col.name, corr = r}
  end

  -- Sort by most negative correlation (strongest negative link to MPG)
  table.sort(corrs, function(a, b)
    return (a.corr or 0) < (b.corr or 0)
  end)

  -- Print top 3 numeric attributes
  for i = 1, math.min(3, #corrs) do
    local c = corrs[i]
    print(string.format("Numeric: %-10s  correlation = %.4f", c.name, c.corr))
  end

  -- For categorical columns, find category most over‑represented
  -- in low‑MPG group
  local cat_cols = {}
  for _, col in ipairs(tbl.cols.all) do
    if col.name ~= "Mpg+" and (col.__index == SYM or col.__index == ezr.SYM) then
      cat_cols[#cat_cols + 1] = col
    end
  end

  local cat_scores = {}
  for _, col in ipairs(cat_cols) do
    local overall = {}
    local low     = {}
    for _, r in ipairs(tbl.rows) do
      local v = r[col.at]
      if v ~= "?" and v ~= nil then
        overall[v] = (overall[v] or 0) + 1
      end
    end
    for _, r in ipairs(low_rows) do
      local v = r[col.at]
      if v ~= "?" and v ~= nil then
        low[v] = (low[v] or 0) + 1
      end
    end
    -- find category with largest absolute difference in proportions
    local best = nil
    for v, lcnt in pairs(low) do
      local ocnt = overall[v] or 0
      local prop_low = lcnt / #low_rows
      local prop_over = ocnt / #tbl.rows
      local diff = math.abs(prop_low - prop_over)
      if not best or diff > best.diff then
        best = {cat = v, diff = diff, prop_low = prop_low, prop_over = prop_over}
      end
    end
    if best then
      cat_scores[#cat_scores + 1] = {
        name = col.name,
        cat  = best.cat,
        diff = best.diff
      }
    end
  end

  table.sort(cat_scores, function(a, b) return a.diff > b.diff end)

  for i = 1, math.min(3, #cat_scores) do
    local c = cat_scores[i]
    print(string.format("Categorical: %-10s  category %s diff = %.4f",
      c.name, c.cat, c.diff))
  end
end

-- ------------------------------------------------------------------
-- 2. CAR-10 : Common traits of low overall performance
-- ------------------------------------------------------------------

local function car10(tbl)
  print("\n== CAR-10 ==")

  -- Desirability: MPG / Weight
  local mpg_col, lbs_col
  for _, col in ipairs(tbl.cols.all) do
    if col.name == "Mpg+" then mpg_col = col end
    if col.name == "Lbs-" then lbs_col = col end
  end
  if not mpg_col or not lbs_col then error("Required columns missing") end

  local desir_vals = {}
  for _, r in ipairs(tbl.rows) do
    local mpg = r[mpg_col.at]
    local lbs = r[lbs_col.at]
    if mpg ~= "?" and lbs ~= "?" and mpg ~= nil and lbs ~= nil then
      desir_vals[#desir_vals + 1] = mpg / lbs
    end
  end

  local desir_med = median(desir_vals)
  local desir_sd  = stddev(desir_vals, desir_med)

  local low_threshold = desir_med - desir_sd
  local low_rows = {}
  for _, r in ipairs(tbl.rows) do
    local mpg = r[mpg_col.at]
    local lbs = r[lbs_col.at]
    if mpg ~= "?" and lbs ~= "?" and mpg ~= nil and lbs ~= nil then
      local d = mpg / lbs
      if d < low_threshold then low_rows[#low_rows + 1] = r end
    end
  end
  if #low_rows == 0 then
    -- fallback to worst 10%
    local n = math.floor(0.1 * #tbl.rows)
    low_rows = {}
    table.sort(tbl.rows, function(a, b)
      local d1 = (a[mpg_col.at] or 0) / (a[lbs_col.at] or 1)
      local d2 = (b[mpg_col.at] or 0) / (b[lbs_col.at] or 1)
      return d1 < d2
    end)
    for i = 1, n do low_rows[#low_rows + 1] = tbl.rows[i] end
  end

  -- For each column, compute low-group mean/mode and overall mean/mode
  local results = {}
  for _, col in ipairs(tbl.cols.all) do
    local low_vals = {}
    local all_vals = {}
    for _, r in ipairs(tbl.rows) do
      local v = r[col.at]
      if v ~= "?" and v ~= nil then
        all_vals[#all_vals + 1] = v
      end
    end
    for _, r in ipairs(low_rows) do
      local v = r[col.at]
      if v ~= "?" and v ~= nil then
        low_vals[#low_vals + 1] = v
      end
    end

    local overall_mean, overall_mode, low_mean, low_mode
    if col.__index == NUM or col.__index == ezr.NUM then
      overall_mean = ezr.sum(all_vals, function(v) return v end) / #all_vals
      low_mean     = ezr.sum(low_vals, function(v) return v end) / #low_vals
    else
      overall_mode = ezr.new(SYM, {has={}})  -- reuse SYM methods
      for _, v in ipairs(all_vals) do overall_mode:add(v) end
      overall_mode = overall_mode:mid()
      low_mode     = ezr.new(SYM, {has={}})
      for _, v in ipairs(low_vals) do low_mode:add(v) end
      low_mode = low_mode:mid()
    end

    table.insert(results, {
      name = col.name,
      overall = overall_mean or overall_mode,
      low = low_mean or low_mode
    })
  end

  -- Print at least three columns
  for i = 1, math.min(3, #results) do
    local r = results[i]
    if type(r.overall) == "number" then
      print(string.format("%-10s  overall=%.2f  low=%.2f",
        r.name, r.overall, r.low))
    else
      print(string.format("%-10s  overall=%s  low=%s",
        r.name, tostring(r.overall), tostring(r.low)))
    end
  end
end

-- ------------------------------------------------------------------
-- Main ---------------------------------------------------------------
-- ------------------------------------------------------------------

local function main(csv_path)
  -- Load whole table
  local tbl = ezr.Tbl(csv(csv_path))

  car09(tbl)
  car10(tbl)
end

if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_05.lua <csv_path>\n")
end
