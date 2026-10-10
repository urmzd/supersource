"""file-produced `rows` (DESIGN 5.7): parquet row counts read from the footers
with pyarrow in the learner's environment; with --smoke the step's
smoke_vars shrink the run (the C1 smoke mode mechanism)."""

WRITER = """import sys
import pyarrow as pa
import pyarrow.parquet as pq
out, rows = sys.argv[1], int(sys.argv[2])
half = rows // 2
for i, n in enumerate((half, rows - half)):
    pq.write_table(pa.table({"text": [f"doc {j}" for j in range(n)]}), f"{out}/shard-{i}.parquet")
print("wrote", rows)
"""


def test_parquet_rows_and_smoke_vars(ss):
    ss.add_extras("rows")
    ss.init()
    ss.install_system()
    (ss.learner / "shards.py").write_text(WRITER)
    st = ss.learner / "system.toml"
    st.write_text(
        st.read_text().replace(
            "[entry]\n",
            '[entry]\nctl = ["uv", "run", "--quiet", "--no-project", "--with", "pyarrow", "python", "shards.py"]\n',
        )
    )
    ss.commit_learner("feat: shard writer")
    out = ss("milestone", "MS-R90", rc=0, timeout=300).out
    assert "PASS" in out
    # --smoke writes 120 rows (smoke_vars), which the full bar rejects
    out = ss("milestone", "MS-R90", "--smoke", rc=1, timeout=300).out
    assert "hold 120 rows; want >= 1200" in out
