import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import chardet
import pytest

DATA = []
FILE_PATH = Path(os.getcwd()) / "row_data.txt"


@pytest.fixture(scope="function", autouse=True)
def set_up_function():
    try:
        yield
    except Exception as ex:
        raise ex
    finally:
        with open(FILE_PATH, "w") as FILE:
            FILE.write("processes lsize msize rsize result\n")
            FILE.write("\n".join(DATA))


def get_cmd(processes: int, lsize: int, msize: int, rsize: int, splitting_type: int):
    return f"docker run --rm --privileged hw/mpi:latest " \
           f"mpirun -n {processes} --oversubscribe /app/main {lsize} {msize} {rsize} {splitting_type}"


def get_time_from_output(output: str) -> Optional[float]:
    search_result = re.search(r'Time = ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)', output)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def run_cmd(cmd: str) -> str:
    output = subprocess.check_output(cmd, shell=True)
    encoding = chardet.detect(output)['encoding']

    return output.decode(encoding)


num_of_proc_list = [1, 2, 3, 4, 5, 6, 7, 8]
mat_size_list = [8, 64, 256, 1024]
# num_of_proc_list = [8]
# mat_size_list = [256]


class TestClass:

    @pytest.mark.parametrize("num_of_proc", num_of_proc_list)
    @pytest.mark.parametrize("lsize", mat_size_list)
    @pytest.mark.parametrize("msize", mat_size_list)
    @pytest.mark.parametrize("rsize", mat_size_list)
    def test_my_run(self, num_of_proc, lsize, msize, rsize):
        cmd = get_cmd(num_of_proc, lsize, msize, rsize, 0)
        print(f"\nStart cmd \"{cmd}\"\n")

        output = run_cmd(cmd)

        global DATA
        result = f"{num_of_proc} {lsize} {msize} {rsize} {get_time_from_output(output)}"
        print(f"\n{result}\n")
        DATA.append(result)
