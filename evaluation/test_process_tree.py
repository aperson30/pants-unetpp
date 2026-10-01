"""Real POSIX descendant-timeout test, with no ML package dependencies."""
import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == 'posix', 'requires POSIX process groups')
class ProcessTreeTest(unittest.TestCase):
    def test_timeout_kills_grandchild_before_return(self):
        # Exercise the actual function's source without importing NumPy/nnU-Net.
        tree = ast.parse(Path(__file__).with_name('predict_and_shrink.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'run_prediction_command')
        namespace = {'os': os, 'signal': signal, 'subprocess': subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<runner>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / 'grandchild.pid'
            code = ('import subprocess,sys,time; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(60)')
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    namespace['run_prediction_command']([sys.executable, '-c', code, str(pid_path)], timeout=2)
                pid = int(pid_path.read_text())
                for _ in range(50):
                    stat = Path(f'/proc/{pid}/stat')
                    if not stat.exists() or stat.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.02)
                else:
                    self.fail('grandchild still running after timeout cleanup')
            finally:
                if pid_path.exists():
                    try: os.kill(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError: pass


if __name__ == '__main__':
    unittest.main()
