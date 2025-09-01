def fmt_table(rows: List[List], title: str) -> str:
    if not rows: return f"*{title}*: _None_\n"
    pretty = [
        [
            r[0],
            m_dollars_int(r[1]),              # F M$
            m_dollars_int(r[2]),              # S M$
            pct_with_emoji(r[3]),             # % 24h
            pct_with_emoji(r[4]),             # %4H
        ] for r in rows
    ]
    return (
        f"*{title}*:\n"
        "```\n" + tabulate(pretty,
            headers=["SYM","F","S","%","%4H"],
            tablefmt="github"
        ) + "\n```\n"
    )

def fmt_table_single(sym: str, fut_usd: float, spot_usd: float, pct: float,
                     pct4h: float, title: str) -> str:
    row = [[sym.upper(),
            m_dollars_int(fut_usd),
            m_dollars_int(spot_usd),
            pct_with_emoji(pct),
            pct_with_emoji(pct4h)]]
    return (
        f"*{title}*:\n"
        "```\n" + tabulate(row, headers=["SYM","F","S","%","%4H"], tablefmt="github") + "\n```\n"
    )
