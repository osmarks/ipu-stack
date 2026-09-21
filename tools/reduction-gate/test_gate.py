"""Integration checks use disposable repositories, never the user's index."""

import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate", ROOT / "check.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
BINARY = ROOT.parent.parent / "target/reduction-gate/release/ipu-reduction-metrics"


class Rules(unittest.TestCase):
    def test_ast_counts_ignore_tests_but_keep_platform_code(self):
        source = '''
struct Pair(u32, u32);
enum Choice {
    Empty,
    Value { payload: Pair },
    #[cfg(test)]
    TestOnly(u32),
}
trait Interface {
    type Output;
    fn method(&self);
}
impl Interface for Pair {
    type Output = Pair;
    fn method(&self) { let _ = || 1; }
}
#[cfg(any(test, target_os = "linux"))]
fn platform() {}
#[cfg(all(test, feature = "extra"))]
mod tests { struct Hidden { field: u32 } fn test() {} }
#[test]
fn test_function() {}
'''
        result = subprocess.check_output([BINARY], input=json.dumps({"path": "x.rs", "source": source}), text=True)
        counts = json.loads(result)
        self.assertEqual({k: counts[k] for k in gate.METRICS[1:]},
                         dict(types=5, variants=2, fields=3, functions=3))


class IndexChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.env = os.environ | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        self.git("init", "-q")
        self.git("config", "user.name", "Gate test")
        self.git("config", "user.email", "gate@example.invalid")
        # Reuse the already-built counter without rebuilding dependencies for
        # every disposable repository. Cargo still validates the locked build.
        (self.repo / "target").symlink_to(ROOT.parent.parent / "target", target_is_directory=True)
        self.write("crates/demo/src/lib.rs", "struct Removed;\nfn retained() {}\n")
        self.git("add", "crates")
        self.git("commit", "-qm", "fixture baseline")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, env=self.env, text=True)

    def write(self, name, source):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    def check(self, expected, *args):
        result = subprocess.run(["python3", str(ROOT / "check.py"), *args],
                                cwd=self.repo, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_staged_reduction_ignores_unstaged_growth_and_rename(self):
        self.write("crates/demo/src/lib.rs", "fn retained() {}\n")
        self.git("add", "crates")
        self.write("crates/demo/src/lib.rs", "struct Growth;\n" * 20)
        self.check(0)
        self.git("restore", "crates/demo/src/lib.rs")
        self.git("mv", "crates/demo/src/lib.rs", "crates/demo/src/renamed.rs")
        self.check(0)

    def test_unstaged_reduction_reports_no_change(self):
        self.write("crates/demo/src/lib.rs", "")
        self.assertEqual(self.check(0).count("(+0)"), len(gate.METRICS))

    def test_installed_hook_allows_growth_and_reports_it(self):
        (self.repo / "tools").mkdir()
        (self.repo / "tools/reduction-gate").symlink_to(ROOT, target_is_directory=True)
        (self.repo / "scripts").mkdir()
        scripts = ROOT.parent.parent / "scripts"
        shutil.copy(scripts / "module-map.py", self.repo / "scripts")
        shutil.copytree(scripts / "module-map", self.repo / "scripts/module-map")
        self.git("config", "core.hooksPath", str(ROOT.parent.parent / ".githooks"))
        self.write("crates/demo/src/lib.rs", "struct Removed;\nfn retained() {}\nstruct Added { value: u32 }\n")
        self.git("add", "crates")
        result = subprocess.run(["git", "commit", "-m", "growth is advisory"],
                                cwd=self.repo, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Advisory only", result.stdout + result.stderr)
        self.assertIn("(+1)", result.stdout + result.stderr)

    def test_invalid_rust_warns_without_blocking(self):
        self.write("crates/demo/src/lib.rs", "fn incomplete(\n")
        self.git("add", "crates")
        self.assertIn("advisory source metrics unavailable", self.check(0))

    def test_macro_change_reports_measurement_limit(self):
        self.write("crates/demo/src/lib.rs", "macro_rules! hidden { () => { struct New; } }\n")
        self.git("add", "crates")
        self.assertIn("macro definitions/includes changed", self.check(0))


if __name__ == "__main__":
    subprocess.run(["cargo", "build", "--quiet", "--release", "--locked",
                    "--manifest-path", str(ROOT / "Cargo.toml"),
                    "--target-dir", str(BINARY.parents[1])], check=True)
    unittest.main()
