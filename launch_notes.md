## Test (public docs demo)

./dev/run_model_server.sh --config server/model_server/config.yaml --cuda-devices 0 --warmup

MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
  MOLSYS_AI_PROJECT_INDEX_DIR=server/chat_api/data/indexes \
  MOLSYS_AI_EMBEDDINGS=sentence-transformers \
  MOLSYS_AI_CORS_ORIGINS=https://www.uibcdf.org \
  ./dev/run_chat_api.sh --host 127.0.0.1 --port 8000

curl https://api.uibcdf.org/healthz

Open the published pilot:

- https://www.uibcdf.org/molsys-ai-server/

curl -sS -X POST https://api.uibcdf.org/v1/chat \
    -H 'Content-Type: application/json' \
    -d '{
      "messages":[{"role":"user","content":"What is MolSysMT? Give a short answer and cite sources."}],
      "rag":"on",
      "sources":"on",
      "k":5
    }'


{"answer":"MolSysMT is a project conceived and developed by the members of the Computational Biology and Drug Design Research Unit (UIBCDF) at the Mexican National Institute of Health - Children's Hospital of Mexico Federico Gómez [1]. The exact purpose or function of MolSysMT is currently not specified in the available documentation [2]. \n\nPlease consult the Quickstart guide for a first contact with the tool [5], or the tutorial [4] for more information on using MolSysMT.","sources":[{"id":1,"path":"molsysmt/docs/content/about/who.md","section":"# Who is behind?","label":null,"url":"https://www.uibcdf.org/molsysmt/content/about/who.html"},{"id":2,"path":"molsysmt/docs/content/user/intro/molsysmt.ipynb","section":"# What's MolSysMT?","label":null,"url":"https://www.uibcdf.org/molsysmt/content/user/intro/molsysmt.html"},{"id":3,"path":"molsysmt/api_surface/form/molsysmt_GROFileHandler/to_molsysmt_MolSys.md","section":null,"label":null,"url":null},{"id":4,"path":"molsysmt/recipes/notebooks_tutorials/docs/content/user/intro/molsysmt/tutorial.md","section":"# Tutorial (notebook)","label":null,"url":"https://www.uibcdf.org/molsysmt/content/user/intro/molsysmt/tutorial.html"},{"id":5,"path":"molsysmt/recipes/notebooks_tutorials/docs/content/user/index/tutorial.md","section":"## Quickstart guide","label":null,"url":"https://www.uibcdf.org/molsysmt/content/user/index/tutorial.html"}],"debug":null}


