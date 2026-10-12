# Collecting a real Manipuri textile dataset

The procedural grammar teaches the model *structure*; authenticity has to come from real
textiles. A modest, carefully documented dataset (a few hundred pieces) is already a research
contribution, because no public dataset of Manipuri textile designs exists.

## 1. Consent and partnership first

* Work with weavers, cooperatives, shop owners, museums or families who own the pieces. Explain
  the project in plain language (Meiteilon/Manipuri and English), what will be published (design
  grids, photographs, a demo) and how they will be credited.
* Record **informed consent** per piece or per collection (template below). Respect "no" and
  "photo only, do not publish".
* Prefer pieces whose maker or source is known; record it if the owner agrees.
* Agree on how the community benefits (credit in the paper and demo, sharing generated designs
  and weaving booklets, co-authorship or acknowledgement for expert contributors).
* *Moirang Phee* (and possibly other Manipuri textiles) are registered Geographical Indications;
  research use is generally fine, but do not market generated designs under these names.

## 2. Photography protocol

| Do | Avoid |
|---|---|
| Lay the cloth **flat**, fully stretched, on a plain contrasting background | Folds, draping, worn garments |
| Diffuse daylight (shade / overcast) or two lamps at 45° | Direct sun, flash, coloured light |
| Camera **parallel** to the cloth, filling the frame | Strong angles (some is fine: mark corners later) |
| One overall photo + close-ups of borders and motifs | Only close-ups |
| A ruler (cm) and, if available, a colour checker in a corner | Filters, beautification, HDR |
| Highest resolution, JPEG quality high (or RAW) | Messaging-app compression |

For a 1 m wide cloth a 12-MP phone photo gives roughly 4 px per thread — enough. For fine motifs
take close-ups at ≥ 10 px per motif cell.

## 3. Folder layout and metadata

```
data/raw/<piece_id>/
    overall.jpg, border_1.jpg, motif_1.jpg, ...
    meta.json
```

`meta.json` (one per piece):

```json
{
  "piece_id": "MP-0001",
  "category": "moirang_phee",
  "local_name": "Moirang Phee",
  "garment": "innaphi | phanek | shawl | ...",
  "region": "Moirang, Bishnupur district",
  "maker": "name or 'unknown' (only with consent)",
  "owner": "cooperative / family / museum",
  "year_made": "approx. or unknown",
  "loom": "loin loom | throw-shuttle frame loom | fly-shuttle | jacquard | unknown",
  "technique": "supplementary weft | warp stripes | weft stripes | ...",
  "fibre": "cotton | silk | mixed | unknown",
  "consent": {"photo": true, "publish_photo": false, "publish_design": true, "date": "2026-11-02"},
  "photographer": "...",
  "notes": "..."
}
```

Suggested categories (confirm names and boundaries with weavers / textile experts):
`phanek_mayek_naibi`, other phanek types, `moirang_phee`, `leirum_phee`, `innaphi`,
`wangkhei_phee`, `lasingphee`, and `other`. The code treats categories as free-text labels.

## 4. Digitisation and curation

```bash
manitex digitise data/raw/MP-0001/border_1.jpg --corners x1,y1,x2,y2,x3,y3,x4,y4 \
        --style moirang_phee --out data/designs/MP-0001
```

1. Check `*_grid.png`: magenta lines should follow the motif steps. If not, count the cells across
   (and down) a motif and pass `--cells-x/--cells-y`.
2. Check `*_preview.png` against the photo; fix small errors in any pixel editor (the `.png` is an
   indexed image, so the palette is preserved) or re-run with `--colors N`.
3. Keep only designs you would sign off as faithful. Record digitisation problems in
   `meta.json` — they are useful for the paper's error analysis.
4. Optionally have a second person check each design (inter-annotator agreement).

## 5. Splits and shards

Split by **piece** (never put two crops of the same textile in different splits), stratified by
category: e.g. 80 / 10 / 10 % train / validation / test.

```bash
manitex data import data/designs/train --out data/real/train.npz
manitex data import data/designs/val   --out data/real/val.npz
manitex data import data/designs/test  --out data/real/test.npz
```

## 6. Data card (fill in for the paper)

* Size: number of pieces / photos / designs per category; grid size statistics.
* Provenance: locations, owners, period of collection, photographers.
* Consent: what may be published; how withdrawals are handled.
* Known biases: e.g. market pieces vs heirlooms, modern synthetic dyes vs traditional dyes.
* Licence: which parts (if any) are released, under which terms.

## Consent form template (adapt and translate)

> I agree that photographs of the textile(s) listed below may be taken by ____ for the research
> project "manitex — computational study of Manipuri textile patterns" at ____.
>
> ☐ The photographs may be used for research analysis only.
> ☐ The design (pattern grid) may be published in a research paper / open dataset / online demo.
> ☐ The photographs may be published.
> ☐ The maker / owner may be credited as: ____ ☐ Please keep the source anonymous.
>
> I can withdraw this consent at any time by contacting ____ ; published material will then be
> removed from future versions.
>
> Name, signature, date.
