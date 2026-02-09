import glob
import subprocess

if __name__ == "__main__":
    without_file = ["main.py", "create_pyi.py", "setup.py"]
    directory = ["./"]
    for dr in directory:
        files = glob.glob(f"{dr}/*.py")
        for file_path in files:
            if file_path.endswith(".py"):
                flag = True
                for one in without_file:
                    if one in file_path:
                        flag = False
                        continue
                if flag:
                    subprocess.run(["stubgen", file_path, "-o", "./", "--include-docstrings", "--include-private"])
