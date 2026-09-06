#!/usr/bin/env python3
"""Probe가 남긴 trace JSON을 템플릿에 박아 자체 완결 HTML 한 장으로 만든다.

    mvn exec:java -Dexec.mainClass=lab.resnet.Probe
    python3 tools/viz/build.py

두 벌을 뽑는다.

    docs/index.html       GitHub Pages가 그대로 서빙하는 완전한 문서
    build/viz/cycle.html  <head> 없는 조각. 문서 골격을 스스로 씌우는 뷰어용

외부 요청이 하나도 없어야 하므로 JSON을 파일로 두지 않고 인라인한다.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tools" / "viz" / "cycle.template.html"
TRACES = ROOT / "build" / "probe"
DOCS = ROOT / "docs" / "index.html"
FRAGMENT = ROOT / "build" / "viz" / "cycle.html"

# 템플릿은 <title>과 <style>로 시작하는 조각이다. 여기서 문서 골격을 씌운다.
HEAD = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<meta property="og:title" content="이미지 한 장이 ResNet을 한 바퀴 도는 동안">
<meta property="og:description" content="{description}">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,\
%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E\
%3Ctext y='14' font-size='14'%3E%F0%9F%94%AC%3C/text%3E%3C/svg%3E">
"""
DESCRIPTION = (
    "CIFAR-10 이미지 한 장이 ResNet-20의 순전파·손실·역전파·가중치 갱신을 "
    "한 바퀴 도는 동안 실제로 나온 텐서 값을 단계별로 펼쳐 본 계측 기록."
)


def load(version):
    path = TRACES / f"trace-{version}.json"
    if not path.exists():
        sys.exit(f"{path} 가 없다. 먼저 Probe를 돌릴 것:\n"
                 f"  mvn exec:java -Dexec.mainClass=lab.resnet.Probe")
    text = path.read_text(encoding="utf-8")
    json.loads(text)  # 깨진 JSON을 조용히 심지 않도록 여기서 한 번 검증한다
    # </script> 가 들어 있으면 인라인 script 블록이 거기서 끊긴다.
    return text.replace("</", "<\\/")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{path.relative_to(ROOT)}  ({len(text.encode('utf-8')):,} bytes)")


def main():
    fragment = TEMPLATE.read_text(encoding="utf-8")
    for version in ("v1", "v2"):
        fragment = fragment.replace(f"__TRACE_{version.upper()}__", load(version))

    # <title>은 <head> 안에 있어야 한다. 조각 맨 앞의 한 줄을 잘라 옮긴다.
    title, _, body = fragment.partition("\n")
    assert title.startswith("<title>"), "템플릿 첫 줄은 <title>이어야 한다"

    write(DOCS, HEAD.format(description=DESCRIPTION) + title + "\n</head>\n<body>\n"
          + body.lstrip() + "\n</body>\n</html>\n")
    write(FRAGMENT, fragment)

    nojekyll = DOCS.parent / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.touch()
        print(f"{nojekyll.relative_to(ROOT)}  (생성)")


if __name__ == "__main__":
    main()
