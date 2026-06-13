-- table-widths.lua — give every table proportional column widths so wide
-- cells wrap instead of overflowing the page margin.
--
-- GFM pipe tables arrive with no column widths, so pandoc's LaTeX writer emits
-- non-wrapping `l` columns and long cells run off the page. This filter sizes
-- each column to its longest cell (as a fraction of the row), which makes the
-- writer emit wrapping `p{width}` columns.

local stringify = pandoc.utils.stringify

local function cell_length(cell)
  return math.max(1, #stringify(cell.contents))
end

local function scan(rows, maxw)
  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      local len = cell_length(cell)
      if not maxw[i] or len > maxw[i] then
        maxw[i] = len
      end
    end
  end
end

function Table(tbl)
  local ncol = #tbl.colspecs
  if ncol == 0 then
    return nil
  end

  local maxw = {}
  scan(tbl.head.rows, maxw)
  for _, body in ipairs(tbl.bodies) do
    scan(body.body, maxw)
  end

  local total = 0
  for i = 1, ncol do
    maxw[i] = maxw[i] or 1
    total = total + maxw[i]
  end

  -- Reserve nothing; let the writer scale to \linewidth. Clamp tiny columns so
  -- single-character cells still get a usable minimum.
  for i = 1, ncol do
    local frac = maxw[i] / total
    tbl.colspecs[i][2] = math.max(frac, 0.06)
  end

  return tbl
end
