
## Usage

```bash
cd /path/to/wedefense/docs
pip install -r docs-requirements.txt
make clean
sphinx-build -M html source build
cd build/html
python3 -m http.server 8000
```

You should see:

```nginx
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

Open your browser and go to <http://0.0.0.0:8000/> to view the generated
documentation.

Done!

**Hint**: You can change the port number when starting the server. For example `python3 -m http.server 8888`
