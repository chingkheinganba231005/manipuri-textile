# From design to loom: representation and weave compiler

## Conventions

| Item | Convention |
|---|---|
| Design grid | `grid[row, col]` = palette index of one **design cell**; `row 0` is the top of the image. |
| Face | The image shows the **face** (right side) of the cloth. |
| Columns | Column `j` → warp ends `j·e … j·e+e−1` (`e` = `ends_per_cell`). End 1 is the **leftmost** end seen from the face. |
| Rows | Weaving starts at the **bottom** image row; chart row 1 = bottom row; pick 1 = first pick woven. (`bottom_up=False` reverses this.) |
| Lift plan | `liftplan[p, e] = 1` → end `e` is **raised** on pick `p`: the warp lies on top of the weft and shows on the face. |
| Colours | Canonical order: index 0 is the most frequent colour (normally the ground). |

## The draft (`WeaveDraft`)

A draft stores the lift plan (picks × ends), the colour of every warp end and weft pick, the
kind of every pick (ground / supplementary), the design row each pick belongs to and the design
column of each end. Everything else is **derived**:

* **Threading** — one shaft per distinct end behaviour (distinct lift-plan columns), numbered by
  first use from the left. Reconstruction is exact: `shaft_lifts[:, threading] == liftplan`.
* **Tie-up / treadling** — one treadle per distinct shaft-lift row, single treadle per pick
  (rising shed).
* **Floats** — longest runs of the weft over/under the warp (face/back) and of the warp.
* **Face appearance** — warp colour where raised, weft colour elsewhere (inlay-aware).

## Brocade technique (default)

Most Manipuri motif work — e.g. the temple row of a Moirang phee border — is woven with an
**extra (supplementary) weft** over a plain ground. The compiler:

1. **Plans the ground.** Which cells can the ground cloth itself show?
   * `none` — one ground colour (most frequent);
   * `weft` — rows that are ≥ 90 % one colour become **weft stripes** (ground picks of that
     colour);
   * `warp` — columns that are ≥ 90 % one colour become **warp stripes**;
   * `plaid` — crossing warp and weft stripes. Colours that look like the optical mixture of two
     others (CIEDE2000 < 10) are recognised as crossings, and the warp / weft colour sequences are
     recovered by voting;
   * `auto` picks the plan that explains most cells (ties: none → weft → warp; plaid only with a
     clear margin).
2. **Emits picks row by row** (bottom-up): `picks_per_cell` ground picks, each followed by one
   supplementary pick per pattern colour in that row (*continuous* weft, selvedge to selvedge,
   floating on the back where hidden) or by a single pick carrying every colour as separate
   bobbins (*discontinuous* inlay).
3. **Binds long floats.** In each visible run longer than `max_float` ends, tie-down ends are
   raised on an absolute lattice `(end − phase) mod max_float = 0` (never on the first or last end
   of a run). The phase alternates by half a step between successive pattern picks (*brick*
   stagger) or advances by one (*twill*). A repair step guarantees every remaining float is
   ≤ `max_float`.
4. **Balances rows.** With `balance_rows=True`, rows without pattern picks get extra ground picks
   so that every design row occupies the same length of cloth (one pattern pass per ground pick
   is the reference). This keeps the woven proportions equal to the design's.

Ground weaves: `plain`, `twill_2_1`, `twill_1_2`, `twill_2_2`, `twill_3_1`, `twill_1_3`, `satin_5_warp`,
`satin_5_weft`, `basket_2`, `warp_rib_2`, `weft_rib_2` (any `twill_U_D[_s]`, `satin_N[_move][_warp|_weft]`
is accepted).

## Damask technique

For two-colour designs: warp = ground colour, weft = figure colour, warp-faced satin where the
ground is and weft-faced satin elsewhere (jacquard). Choose `ends_per_cell = picks_per_cell = 5`
so every cell contains a full satin repeat.

## Outputs

* **WIF 1.1** (`save_wif`): `[THREADING]` (minimal shafts), `[LIFTPLAN]`, optional `[TIEUP]` /
  `[TREADLING]` when the number of distinct sheds is small, colour table, warp/weft colours, notes.
  Readable by most weaving programs; `read_wif` reads WIF files back (lift plan or tie-up +
  treadling, rising or sinking shed, any colour range).
* **Pick-up instructions** — for each supplementary pick, a run-length code such as
  `D8 L1 D7 L4 …`: `L n` = *lift* n ends with the pick-up stick (pattern yarn hidden under them),
  `D n` = leave n ends *down* (pattern yarn floats on top and shows). A lone `L1` inside a long
  `D` run is a tie-down. Identical consecutive rows are grouped.
* **Chart** (`design_chart_svg`) — colour + symbol per cell (prints in black and white), bold
  lines every 10 cells, row numbers from the bottom.
* **Yarn estimate** — per colour: warp ends and length (woven length × (1 + take-up) + loom
  waste), weft picks and length (cloth width × (1 + take-up); inlay picks only cover their region
  plus a turn allowance).
* **Booklet** (`to_html`) — a single printable HTML file with the design, cloth render, overview,
  yarns, warping order, chart and pick-by-pick table.

## Loom notes

* **Loin loom / frame loom with pick-up**: weave the ground with the heddle and shed rod; insert
  each pattern pick with a pick-up stick following the `L/D` codes.
* **Dobby / jacquard**: the instructions report the number of shafts and distinct sheds needed for
  the full design; load the WIF file. If the shaft count exceeds your loom, use pick-up for the
  pattern picks (ground needs only 2 shafts for plain weave), or simplify the design.

## Simulated cloth

`render_draft` draws every crossing as the yarn on top, with cylindrical shading, darkening where
a yarn dives, twist texture and thread irregularity. The ground cloth is drawn continuously
(hidden pattern yarns lie behind it) and visible pattern floats are overlaid, spreading over the
ground picks of their own design row. `face_mode` (`auto`, `balanced`, `weft`, `warp`) controls
whether weft or warp stripes look solid, as in weft- or warp-faced cloth.
