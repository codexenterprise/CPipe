python setup.py build_ext --inplace
python create_pyi.py

mkdir -p ./cpipe_nodes

# move the so file to the current folder
mv ./build/*/*.so ./cpipe_nodes/
mv ./*.pyi ./cpipe_nodes/

python3 -m nuitka --clang --show-memory --show-progress --static-libpython=no --nofollow-imports --output-dir=out main.py
mv ./out/main.bin ./

# delete the build folder
rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf *.c
rm -rf *.so
rm -rf out