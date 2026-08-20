from pydantic import BaseModel, Field


class ProfileData(BaseModel):
    """Структурированные данные профиля, извлекаемые из CV."""

    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    desired_role: str | None = None
    desired_salary: str | None = None
    desired_location: str | None = None
