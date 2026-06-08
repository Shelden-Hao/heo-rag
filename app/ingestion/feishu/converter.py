from langchain_core.documents import Document


def blocks_to_markdown(blocks: list[dict]) -> str:
    lines = []
    for block in blocks:
        block_type = block.get("block_type")
        text = _extract_text(block)
        if not text:
            continue

        if block_type == 3:  # heading1
            lines.append(f"# {text}")
        elif block_type == 4:  # heading2
            lines.append(f"## {text}")
        elif block_type == 5:  # heading3
            lines.append(f"### {text}")
        elif block_type == 6:  # heading4
            lines.append(f"#### {text}")
        elif block_type == 7:  # heading5
            lines.append(f"##### {text}")
        elif block_type == 8:  # heading6
            lines.append(f"###### {text}")
        elif block_type == 9:  # heading7
            lines.append(f"####### {text}")
        elif block_type == 10:  # heading8
            lines.append(f"######## {text}")
        elif block_type == 11:  # heading9
            lines.append(f"######### {text}")
        elif block_type == 12:  # bullet
            lines.append(f"- {text}")
        elif block_type == 13:  # ordered
            lines.append(f"1. {text}")
        elif block_type == 14:  # code
            lang = _extract_code_language(block)
            lines.append(f"```{lang}\n{text}\n```")
        elif block_type == 15:  # quote
            lines.append(f"> {text}")
        elif block_type == 17:  # todo
            done = block.get("todo", {}).get("style", {}).get("done", False)
            checkbox = "[x]" if done else "[ ]"
            lines.append(f"- {checkbox} {text}")
        elif block_type == 22:  # table
            lines.append(text)
        else:
            lines.append(text)

    return "\n\n".join(lines)


def _extract_text(block: dict) -> str:
    for key in ("text", "heading1", "heading2", "heading3", "heading4",
                "heading5", "heading6", "heading7", "heading8", "heading9",
                "bullet", "ordered", "code", "quote", "todo", "equation"):
        body = block.get(key)
        if body and "elements" in body:
            return _parse_elements(body["elements"])
    return ""


def _parse_elements(elements: list[dict]) -> str:
    parts = []
    for elem in elements:
        if "text_run" in elem:
            content = elem["text_run"].get("content", "")
            style = elem["text_run"].get("text_element_style", {})
            if style.get("bold"):
                content = f"**{content}**"
            if style.get("italic"):
                content = f"*{content}*"
            if style.get("strikethrough"):
                content = f"~~{content}~~"
            if style.get("inline_code"):
                content = f"`{content}`"
            if style.get("link") and style["link"].get("url"):
                url = style["link"]["url"]
                content = f"[{content}]({url})"
            parts.append(content)
        elif "mention_user" in elem:
            parts.append("@用户")
        elif "mention_doc" in elem:
            title = elem["mention_doc"].get("title", "文档")
            parts.append(f"[{title}]")
        elif "equation" in elem:
            parts.append(f"${elem['equation'].get('content', '')}$")
    return "".join(parts)


def _extract_code_language(block: dict) -> str:
    code_body = block.get("code", {})
    lang = code_body.get("style", {}).get("language", 0)
    lang_map = {
        1: "python", 2: "java", 3: "javascript", 4: "go",
        5: "c", 6: "cpp", 7: "shell", 8: "sql", 9: "json",
        10: "yaml", 11: "markdown", 12: "html", 13: "css",
    }
    return lang_map.get(lang, "")


def blocks_to_document(blocks: list[dict], title: str, doc_id: str) -> Document:
    markdown = blocks_to_markdown(blocks)
    return Document(
        page_content=markdown,
        metadata={
            "source": f"feishu://{doc_id}",
            "filename": f"{title}.md",
            "doc_id": doc_id,
        },
    )
