# 工具脚本 (scripts)

- **migrate_outputs.py** — 将旧的 `*_output`、`triple_attention_plots` 内容复制到 `output/` 对应子目录（一次性）。
- **merge_and_remove_outputs.py** — 将上述目录合并到 `output/` 后删除旧目录。
- **check_length.py** — 长度/格式检查等辅助脚本。

运行方式（在**仓库根目录**）：  
`python scripts/migrate_outputs.py`  
`python scripts/merge_and_remove_outputs.py`
