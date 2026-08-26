from src.rag.schemas import ProfileData


class ProfileRepository:
    """Доступ к профилю соискателя: upsert структурированных данных CV."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def save(self, user_id: int, data: ProfileData) -> None:
        """Сохранить профиль: обновить существующий или создать новый."""
        from sqlalchemy import select

        from src.database.models import Profile

        async with self._session_factory() as session:
            result = await session.execute(
                select(Profile).where(Profile.user_id == user_id)
            )
            profile = result.scalars().first()
            if profile is None:
                profile = Profile(user_id=user_id)
                session.add(profile)
            profile.skills = data.skills
            profile.experience = data.experience
            profile.desired_role = data.desired_role
            profile.desired_salary = data.desired_salary
            profile.desired_location = data.desired_location
            await session.commit()
