from pydantic import BaseModel


class BaseResponse(BaseModel):
    status: str
    message: str


class PredicaoTurma(BaseModel):
    coddis: str
    codtur: str
    estmtr: int
    capacidade_sugerida: int
    is_calouros: bool


class PredicaoResponse(BaseModel):
    semestre_alvo: int
    total_turmas: int
    predicoes: list[PredicaoTurma]