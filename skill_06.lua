#!/usr/bin/env lua
-- skill_06.lua
-- Run: lua skill_06.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-11 – Explain the effort estimate for a proposed project using historical projects.
--   NASA-12 – Explain the main differences between two proposals and the trade‑offs that drove their risk assessments.

local ezr = require "ezr"

-- Helpers from ezr
local csv      = ezr.csv
local Tbl      = ezr.Tbl
local push     = ezr.push
local sorted   = ezr.sorted
local cap      = math.floor
local sqrt     = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_06.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Column indices (as per the dataset)
-- ------------------------------------------------------------------
local id_at      = 1
local size_at    = 26   -- Kloc+
local effort_at  = 27   -- Effort-
local defect_at  = 28   -- Defects-
local acap_at    = 17   -- acap
local cplx_at    = 11   -- cplx

-- ------------------------------------------------------------------
-- Utility helpers -----------------------------------------------
-- ------------------------------------------------------------------
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  return {mean = col:mid(), std = col:div()}
end

-- distance between two rows on all feature columns (excluding goals)
local function row_dist(row1, row2)
  return tbl:distx(row1, row2)
end

-- ------------------------------------------------------------------
-- NASA-11: Explain effort estimate using historical projects
-- ------------------------------------------------------------------
local effort_stats = col_stats(tbl, effort_at)
local mean_effort  = effort_stats.mean
local std_effort   = effort_stats.std

-- Take the last row as the proposed project
local proposed = tbl.rows[#tbl.rows]

-- Estimate: mean ± std
local est_low  = mean_effort - std_effort
local est_high = mean_effort + std_effort

print("== NASA-11 ==")
print(string.format(
  "Estimated effort for the proposed project (ID=%s) is %.2f ± %.2f man‑months (mean ± std).",
  proposed[id_at], mean_effort, std_effort))
print(string.format(
  "Range of plausible effort: [%.2f, %.2f] man‑months.",
  est_low, est_high))

-- Find the five most similar historical projects
local neighbors = {}
for _, r in ipairs(tbl.rows) do
  if r ~= proposed then
    push(neighbors, {row=r, dist=row_dist(proposed, r)})
  end
end
neighbors = sorted(neighbors, function(a,b) return a.dist < b.dist end)
neighbors = cap(#neighbors, 5)

print("\nTop 5 most similar historical projects:")
for i, n in ipairs(neighbors) do
  print(string.format(
    "  %d) ID=%s | Effort=%.1f | Size=%.1f | ACAP=%s | CPLX=%s | Dist=%.3f",
    i,
    n.row[id_at],
    n.row[effort_at],
    n.row[size_at],
    n.row[acap_at],
    tostring(n.row[cplx_at]),
    n.dist))
end

-- ------------------------------------------------------------------
-- NASA-12: Explain differences between two proposals
-- ------------------------------------------------------------------
local prop1 = tbl.rows[1]
local prop2 = tbl.rows[2]

-- Risk score as z‑score of effort relative to historical mean
local function z_score(val, mean, std)
  return (val - mean) / (std + 1e-12)
end

local risk1 = z_score(prop1[effort_at], mean_effort, std_effort)
local risk2 = z_score(prop2[effort_at], mean_effort, std_effort)

print("\n== NASA-12 ==")
print(string.format(
  "Proposal 1 (ID=%s) risk = z = %.2f (higher is riskier).",
  prop1[id_at], risk1))
print(string.format(
  "Proposal 2 (ID=%s) risk = z = %.2f.",
  prop2[id_at], risk2))

-- Compare key characteristics
local function diff_str(label, v1, v2)
  if type(v1) == "number" and type(v2) == "number" then
    return string.format("%s: %.2f vs %.2f (Δ=%.2f)", label, v1, v2, v1 - v2)
  else
    return string.format("%s: '%s' vs '%s'", label, tostring(v1), tostring(v2))
  end
end

print("\nMain differences affecting risk assessment:")
print(diff_str("Size (Kloc+)", prop1[size_at], prop2[size_at]))
print(diff_str("Effort (man‑months)", prop1[effort_at], prop2[effort_at]))
print(diff_str("ACAP (programmer capability)", prop1[acap_at], prop2[acap_at]))
print(diff_str("Complexity (cplx)", prop1[cplx_at], prop2[cplx_at]))

print("\nTrade‑off insight:")
if prop1[effort_at] > prop2[effort_at] then
  print("  Proposal 1 has higher effort, likely due to larger size or higher complexity.")
else
  print("  Proposal 2 has higher effort, likely due to larger size or higher complexity.")
end
if prop1[acap_at] ~= prop2[acap_at] then
  print("  Programmer capability differences may offset effort differences: higher ACAP can reduce effort.")
end
print("Risk assessment is a balance of size, complexity, and capability; projects with high effort but strong ACAP may still be acceptable.")
