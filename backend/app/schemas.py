"""Contrato da API (validação de entrada e formato de saída)."""
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
