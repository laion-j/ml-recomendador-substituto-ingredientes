"""Contrato da API (validação de entrada e formato de saída)."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BuscaIn(BaseModel):
    receita_texto: str
    ingrediente_faltante: str
    k: int = Field(default=30, ge=1, le=100)  # 1..100: limita o custo do recall

    @field_validator("receita_texto", "ingrediente_faltante")
    @classmethod
    def nao_vazio(cls, v: str) -> str:
        v = v.strip()  # só espaços conta como vazio
        if not v:
            raise ValueError("não pode ser vazio")
        return v


class Candidato(BaseModel):
    nome: str
    categoria: str | None
    funcoes: list[str]
    sabores: list[str]
    distancia: float  # cosseno (<=>): 0 = idêntico, quanto menor mais parecido
    embedding: list[float]  # 384 dims; o front monta o tf.tensor2d


class BuscaOut(BaseModel):
    receita_embedding: list[float]
    ingrediente_embedding: list[float]
    candidatos: list[Candidato]


class ReceitaResumo(BaseModel):
    id: int
    titulo: str
    ingredientes: list


class ParamsTreino(BaseModel):
    """Parâmetros do laboratório. Limites existem para um clique errado não travar a máquina por horas."""
    camadas: list[int] = Field(default=[256, 128, 64], min_length=1, max_length=6)   # neurônios por camada oculta
    ativacao: Literal["relu", "tanh", "elu"] = "relu"
    dropout: float = Field(default=0.3, ge=0, le=0.8)
    l2: float = Field(default=0, ge=0, le=0.05)
    otimizador: Literal["adam", "rmsprop", "sgd"] = "adam"
    taxa: float = Field(default=0.001, ge=1e-5, le=0.1)
    lote: int = Field(default=256, ge=16, le=2048)
    epocas: int = Field(default=60, ge=1, le=300)
    paciencia: int = Field(default=5, ge=0, le=50)      # 0 = early stopping desligado
    restaurar_melhor: bool = True
    neg_por_pos: int = Field(default=5, ge=1, le=15)
    usar_externos: bool = False     # inclui os pares da Recipe1MSubs no treino (nunca na validação)
    seed: int = Field(default=42, ge=0, le=999999)

    @field_validator("camadas")
    @classmethod
    def tamanho_das_camadas(cls, v: list[int]) -> list[int]:
        if any(n < 4 or n > 1024 for n in v):
            raise ValueError("cada camada precisa ter entre 4 e 1024 neurônios")
        return v


class SalvarIn(BaseModel):
    nome: str = Field(min_length=1, max_length=60)
