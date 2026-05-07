################ install uv ################ 
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

################ create environment ################ 
//create a root directory(once can)
mkdir -p ~/uv-envs
//create a python environment and specify python version
uv venv ~/uv-envs/py312 --python 3.12
//manually activate environment
source ~/uv-envs/py312/bin/activate
//exit environment
deactivate

################ set source ################ 
//open file
vim ~/.config/uv/uv.toml
//input the following content
[[index]]
url = "https://mirrors.aliyun.com/pypi/simple"
default = true

################ install cpipe dependencies ################ 
// if dependencies come from requirements.txt  (exported by pip freeze > requirements.txt)
uv pip install -r /path/to/requirements.txt

################ uv export requirements.txt to others  ################ 
uv pip freeze > requirements.txt

################ check environment size ################ 
du -sh ~/uv-envs