def greet(name: str) -> str:
    """Return a greeting, using qiuqiuqiu for blank input.

    空输入会使用 qiuqiuqiu 作为默认问候对象。
    """
    cleaned = name.strip()
    if not cleaned:
        cleaned = "qiuqiuqiu"
    return f"Hello, {cleaned}!"
