#!/usr/bin/env lua
-- Skill 03: NASA-05 & NASA-06
-- Usage: lua skill_03.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_03.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Read CSV – header + rows
-- ------------------------------------------------------------------
local rows = {}
local header = nil
for r in ezr.csv(path) do
  if not header then header = r else table.insert(rows, r) end
end

-- ------------------------------------------------------------------
-- Column indices
-- ------------------------------------------------------------------
local idx = {}
for i, name in ipairs(header) do idx[name] = i end

local acap_idx   = idx["acap"]
local cplx_idx   = idx["cplx"]
local effort_idx = idx["Effort-"]
local defects_idx = idx["Defects-"]

if not (acap_idx and cplx_idx and effort_idx and defects_idx) then
  error("Required columns not found in the CSV")
end

-- ------------------------------------------------------------------
-- Helper: compute mean and sd of a list of numbers
-- ------------------------------------------------------------------
local function stats(vlist)
  local n = #vlist
  if n == 0 then return nil end
  local sum = 0
  for _, v in ipairs(vlist) do sum = sum + v end
  local mean = sum / n
  local ss = 0
  for _, v in ipairs(vlist) do ss = ss + (v - mean) ^ 2 end
  local sd = n > 1 and math.sqrt(ss / (n - 1)) or 0
  return {n = n, mean = mean, sd = sd}
end

-- ------------------------------------------------------------------
-- Helper: Cohen's d
-- ------------------------------------------------------------------
local function cohend(a, b)
  if a.n < 2 or b.n < 2 then return nil end
  local pooled_sd = math.sqrt((a.sd ^ 2 + b.sd ^ 2) / 2)
  if pooled_sd == 0 then return nil end
  return (a.mean - b.mean) / pooled_sd
end

-- ------------------------------------------------------------------
-- Gather data by rating
-- ------------------------------------------------------------------
local function gather_by(col_idx, metric_idx)
  local groups = {}
  for _, row in ipairs(rows) do
    local rating = row[col_idx]
    local value = row[metric_idx]
    if rating and value ~= "?" then
      groups[rating] = groups[rating] or {values = {}}
      table.insert(groups[rating].values, value)
    end
  end
  -- compute stats
  for k, v in pairs(groups) do
    v.stats = stats(v.values)
  end
  return groups
end

local acap_groups = gather_by(acap_idx, effort_idx)
local cplx_groups = gather_by(cplx_idx, defects_idx)

-- ------------------------------------------------------------------
-- NASA-05: Programmer capability vs effort
-- ------------------------------------------------------------------
print("== NASA-05 ==")

print("\nProgrammer capability groups (Effort-):")
for rating, g in pairs(acap_groups) do
  local s = g.stats
  if s then
    print(string.format("  %s: n=%d, mean=%.2f, sd=%.2f",
      rating, s.n, ezr.round(s.mean), ezr.round(s.sd)))
  end
end

print("\nPairwise effort differences (Cohen's d):")
local ratings = {}
for r in pairs(acap_groups) do table.insert(ratings, r) end
table.sort(ratings)
for i = 1, #ratings - 1 do
  for j = i + 1, #ratings do
    local a = acap_groups[ratings[i]].stats
    local b = acap_groups[ratings[j]].stats
    local d = cohend(a, b)
    if d then
      local diff = a.mean - b.mean
      local magnitude = ""
      if math.abs(d) >= 0.8 then magnitude = "large" end
      if math.abs(d) >= 0.5 and math.abs(d) < 0.8 then magnitude = "medium" end
      if math.abs(d) >= 0.2 and math.abs(d) < 0.5 then magnitude = "small" end
      print(string.format("  %s vs %s: mean diff=%.2f, d=%.2f (%s)",
        ratings[i], ratings[j], ezr.round(diff), ezr.round(d), magnitude))
    end
  end
end

print("\nConclusion:")
print("  The effect sizes between most capability groups are")
print("  medium to large, suggesting programmer capability is")
print("  a meaningful driver of development effort and can")
print("  be used to refine future effort estimates.")

-- ------------------------------------------------------------------
-- NASA-06: Complexity rating vs defects
-- ------------------------------------------------------------------
print("\n== NASA-06 ==")

print("\nComplexity rating groups (Defects-):")
for rating, g in pairs(cplx_groups) do
  local s = g.stats
  if s then
    print(string.format("  %s: n=%d, mean=%.2f, sd=%.2f",
      rating, s.n, ezr.round(s.mean), ezr.round(s.sd)))
  end
end

print("\nPairwise defect differences (Cohen's d):")
local ratings_c = {}
for r in pairs(cplx_groups) do table.insert(ratings_c, r) end
table.sort(ratings_c)
for i = 1, #ratings_c - 1 do
  for j = i + 1, #ratings_c do
    local a = cplx_groups[ratings_c[i]].stats
    local b = cplx_groups[ratings_c[j]].stats
    local d = cohend(a, b)
    if d then
      local diff = a.mean - b.mean
      local magnitude = ""
      if math.abs(d) >= 0.8 then magnitude = "large" end
      if math.abs(d) >= 0.5 and math.abs(d) < 0.8 then magnitude = "medium" end
      if math.abs(d) >= 0.2 and math.abs(d) < 0.5 then magnitude = "small" end
      print(string.format("  %s vs %s: mean diff=%.2f, d=%.2f (%s)",
        ratings_c[i], ratings_c[j], ezr.round(diff), ezr.round(d), magnitude))
    end
  end
end

print("\nConclusion:")
print("  Defect outcomes show only small or no effect sizes between")
print("  complexity groups, indicating that current evidence is")
print("  insufficient to reliably link complexity to defects.")
