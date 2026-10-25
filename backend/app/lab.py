"""Laboratório de treino: dispara e acompanha o scripts_ml/lab.py (roda no .venv-tf, que tem o TensorFlow).

O backend NÃO treina: cria a pasta do experimento, lança o processo e lê o progresso.json que ele grava.
Assim o treino (pesado, minutos) não trava a API e o TensorFlow não precisa existir neste venv.
Um experimento = uma pasta data/lab/<id>/ (params.json, progresso.json, modelo.h5, tfjs/, log.txt).
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from .schemas import ParamsTreino, SalvarIn

RAIZ = Path(__file__).resolve().parents[2]
PY_TF = Path(os.getenv("LAB_PYTHON", RAIZ / ".venv-tf" / "bin" / "python"))
SCRIPT = RAIZ / "scripts_ml" / "lab.py"
LAB = RAIZ / "data" / "lab"
MODELOS_ATIVOS = RAIZ / "frontend" / "public" / "models"      # o que o app de substituição carrega
ID_VALIDO = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")          # o id vira nome de pasta: nunca aceitar texto livre
EM_ANDAMENTO = ("preparando", "treinando", "avaliando")

router = APIRouter(prefix="/api/lab")
_procs: dict[str, subprocess.Popen] = {}     # id -> processo lançado por esta instância da API
_cache_dados: dict[tuple[int, bool], dict] = {}           # o resumo dos dados só muda se o banco for recarregado


def _cauda(texto: str, n: int = 1200) -> str:
    return texto.strip()[-n:]


def _pasta(id: str) -> Path:
    pasta = LAB / id
    if not ID_VALIDO.match(id) or not pasta.is_dir():
        raise HTTPException(404, "experimento não encontrado")
    return pasta


def _ativo() -> dict | None:
    arq = LAB / "ativo.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None


def _estado(pasta: Path) -> dict:
    """progresso.json + correções que só a API sabe fazer (processo morto, qual modelo está ativo)."""
    e = json.loads((pasta / "progresso.json").read_text(encoding="utf-8"))
    proc = _procs.get(pasta.name)
    if e["status"] in EM_ANDAMENTO and (proc is None or proc.poll() is not None):
        # o processo acabou sem fechar o progresso: caiu, ou a API foi reiniciada no meio do treino
        log = pasta / "log.txt"
        e["status"] = "falhou"
        e["erro"] = e.get("erro") or (_cauda(log.read_text(errors="replace")) if log.exists() else "processo interrompido")
    ativo = _ativo()
    e["ativo"] = bool(ativo and ativo["id"] == pasta.name)
    return e


@router.get("/dados")
async def dados(neg_por_pos: int = Query(5, ge=1, le=15), externos: bool = False):
    """Como os dados foram separados e que exemplos saem deles (a tela 'Dados')."""
    chave = (neg_por_pos, externos)
    if chave not in _cache_dados:
        r = await run_in_threadpool(
            subprocess.run, [str(PY_TF), str(SCRIPT), "dados", "--neg-por-pos", str(neg_por_pos)] + (["--externos"] if externos else []),
            cwd=RAIZ, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise HTTPException(500, _cauda(r.stderr))
        _cache_dados[chave] = json.loads(r.stdout.strip().splitlines()[-1])   # última linha = o JSON
    return _cache_dados[chave]


@router.post("/treinos", status_code=201)
async def iniciar(params: ParamsTreino):
    # Um treino por vez: dois TensorFlow juntos brigam por CPU/memória e os dois ficam lentos.
    if any(p.poll() is None for p in _procs.values()):
        raise HTTPException(409, "já existe um treino em andamento")
    id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    pasta = LAB / id
    pasta.mkdir(parents=True)
    (pasta / "params.json").write_text(params.model_dump_json(), encoding="utf-8")
    (pasta / "progresso.json").write_text(json.dumps(
        {"id": id, "status": "preparando", "params": params.model_dump(), "historico": [], "criado": time.time()}))
    with open(pasta / "log.txt", "w") as log:
        _procs[id] = subprocess.Popen([str(PY_TF), str(SCRIPT), "treinar", str(pasta)],
                                      cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT)
    return {"id": id}


@router.get("/treinos")
async def listar(limite: int = Query(30, ge=1, le=100)):
    """Histórico, do mais recente ao mais antigo (o id começa com a data, então ordenar por nome basta)."""
    pastas = sorted((p for p in LAB.glob("*") if p.is_dir() and ID_VALIDO.match(p.name)), reverse=True)[:limite] if LAB.exists() else []
    return {"experimentos": [_resumo(_estado(p)) for p in pastas if (p / "progresso.json").exists()], "ativo": _ativo()}


def _resumo(e: dict) -> dict:
    """A lista não precisa da curva inteira nem do erro completo: só do que cabe numa linha da tabela."""
    h = e.get("historico", [])
    return {"id": e["id"], "status": e["status"], "params": e["params"], "criado": e.get("criado"),
            "params_totais": e.get("params_totais"), "epocas_rodadas": len(h), "melhor_epoca": e.get("melhor_epoca"),
            "melhor_val_loss": min((x["val_loss"] for x in h), default=None),
            "metricas": e.get("metricas"), "salvo": e.get("salvo"), "ativo": e["ativo"]}


@router.get("/treinos/{id}")
async def detalhar(id: str):
    return _estado(_pasta(id))


@router.post("/treinos/{id}/parar")
async def parar(id: str):
    """Parada gentil: o script confere este arquivo ao fim de cada época, restaura o melhor ponto e avalia."""
    pasta = _pasta(id)
    if _estado(pasta)["status"] not in EM_ANDAMENTO:
        raise HTTPException(409, "este treino já terminou")
    (pasta / "parar").touch()
    return {"ok": True}


@router.post("/treinos/{id}/salvar")
async def salvar(id: str, corpo: SalvarIn):
    """Exporta o modelo para tfjs (fica na pasta do experimento) e confere a paridade Keras x TFJS no Node."""
    pasta = _pasta(id)
    if _estado(pasta)["status"] not in ("concluido", "parado"):
        raise HTTPException(409, "só dá para salvar um treino terminado")
    r = await run_in_threadpool(
        subprocess.run, [str(PY_TF), str(SCRIPT), "exportar", str(pasta), f"--nome={corpo.nome.strip()}"],
        cwd=RAIZ, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise HTTPException(500, _cauda(r.stderr))
    return _estado(pasta)


@router.post("/treinos/{id}/ativar")
async def ativar(id: str):
    """Troca o modelo que o app de substituição usa. O anterior vai para data/lab/_backup_ativo/ (dá para desfazer)."""
    pasta = _pasta(id)
    salvo = _estado(pasta).get("salvo")
    if not salvo or not salvo["paridade_ok"]:
        raise HTTPException(409, "salve o modelo (com paridade OK) antes de ativar")
    backup = LAB / "_backup_ativo" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True)
    for f in MODELOS_ATIVOS.iterdir():
        shutil.copy2(f, backup / f.name)
    for f in MODELOS_ATIVOS.glob("group1-shard*"):
        f.unlink()                                   # senão sobra .bin do modelo antigo
    for f in (pasta / "tfjs").iterdir():
        shutil.copy2(f, MODELOS_ATIVOS / f.name)
    ativo = {"id": id, "nome": salvo["nome"], "em": datetime.now().isoformat(timespec="seconds"), "backup": backup.name}
    (LAB / "ativo.json").write_text(json.dumps(ativo, ensure_ascii=False))
    return ativo


@router.post("/ativo/desfazer")
async def desfazer():
    """Volta ao modelo que estava ativo antes da última ativação."""
    ativo = _ativo()
    if not ativo:
        raise HTTPException(409, "nenhum modelo do laboratório está ativo")
    backup = LAB / "_backup_ativo" / ativo["backup"]
    for f in MODELOS_ATIVOS.glob("group1-shard*"):
        f.unlink()
    for f in backup.iterdir():
        shutil.copy2(f, MODELOS_ATIVOS / f.name)
    (LAB / "ativo.json").unlink()
    return {"ok": True}
