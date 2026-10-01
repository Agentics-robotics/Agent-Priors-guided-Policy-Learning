"""A complete offline example: recompute both main tables from released outcomes."""
from appl_release.paths import project_root
from appl_release.tables import exp1, exp2

if __name__ == "__main__":
    root = project_root()
    for row in exp1(root):
        print(f"{row['label']}: mean OOD {row['mean_OOD']:.2f}%")
    for row in exp2(root):
        print(f"{row['label']}: motion {row['motion_percent']:.1f}%, composition {row['composition'] or 'not applicable'}")
