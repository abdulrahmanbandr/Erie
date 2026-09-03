# Common Voice 21.0 language audit (2026-09-08)

Speakers with both age and gender labels, per age bin, read from the
remote TSVs (`src/audit_languages.py`). No audio downloaded.
f = female, m = male.

| Language | Labeled speakers | teens f/m | fifties f/m | sixties f/m | seventies f/m | Train audio |
|---|---|---|---|---|---|---|
| ca Catalan | 7,034 | 162 / 165 | 766 / 817 | 394 / 477 | 73 / 118 | 31 tars, 42.5 GB |
| fr French | 4,574 | 64 / 318 | 138 / 300 | 65 / 223 | 15 / 77 | 15 tars, 21.9 GB |
| es Spanish | 4,735 | 164 / 326 | 111 / 271 | 27 / 87 | 3 / 10 | – |
| de German | 4,198 | 43 / 392 | 111 / 372 | 52 / 143 | 8 / 52 | 16 tars, 24.2 GB |
| it Italian | 1,994 | 16 / 80 | 66 / 239 | 34 / 105 | 6 / 17 | – |
| ru Russian | 1,215 | 46 / 124 | 6 / 15 | 2 / 8 | 0 / 0 | 1 tar, 1.0 GB |
| pl Polish | 676 | 5 / 46 | 2 / 6 | 1 / 3 | 1 / 0 | – |
| uk Ukrainian | 483 | 25 / 38 | 1 / 7 | 1 / 1 | 0 / 0 | – |

Full per-bin tables are in the script output; middle bins (twenties to
fourties) are plentiful everywhere.

## Reading

- **Catalan is the only corpus that is balanced in both gender and age.**
  Near 1:1 female/male in every bin, and the largest older population by
  far: 1,583 fifties, 871 sixties, 191 seventies. Teens 327.
- German and French have older speakers but are 3:1 to 5:1 male, and
  female teens are scarce (43 in German).
- Spanish has teens but few over sixty. Polish and Ukrainian are too
  small for the tails.
- Russian is the weakest of the large corpora for the age tails, which
  is what the V3 runs showed.

## Decision

Primary addition: **Catalan**. Second, for language diversity so the
model cannot learn "language = age": **German** or **French**.

Rule for combining languages: every age bin must be represented in every
training language. If older speakers come only from Catalan and teens
only from Russian, a language classifier becomes an age classifier.
Cap speakers per (language, age bin, gender) cell when building the set.

Download cost: Catalan train is 31 shards of ~1.4 GB. Only the ~7k
labeled speakers are needed out of 35k, so the download must go shard by
shard, keep the selected clips, and delete the shard. `data_prep.py`
currently fetches only shard 0 and needs that loop.
