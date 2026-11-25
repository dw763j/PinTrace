"""Update labels in the submitted vector overview while retaining its artwork."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ByteStringObject, ContentStream, FloatObject

ROOT = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
ORIGINAL = ROOT / 'inputs/artwork/pipeline.pdf'
ORIGINAL_SHA256 = 'b2b2adc54635936eae59c91a923345b324325252b184a82cd1fd7da66a0dc8f5'
STEM = 'fig1_evaluation_framework'


def executable(name, poppler_bin):
    if poppler_bin:
        path = Path(poppler_bin) / (name + ('.exe' if os.name == 'nt' else ''))
        if path.is_file():
            return str(path)
    path = shutil.which(name)
    if path:
        return path
    raise FileNotFoundError(f'{name} is required; supply --poppler-bin or add Poppler to PATH.')


def build_overview(output, poppler_bin=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    poppler_bin = poppler_bin or os.environ.get('POPPLER_BIN')
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA256
    reader = PdfReader(ORIGINAL)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    form = writer.pages[0]['/Resources']['/XObject']['/Form1'].get_object()
    font = form['/Resources']['/Font']['/FT9'].get_object()
    cmap = font['/ToUnicode'].get_data().decode('ascii')
    pairs = re.findall(r'<([0-9A-F]+)>\s*<([0-9A-F]+)>',
                       cmap.split('beginbfchar')[1].split('endbfchar')[0])
    decode = {bytes.fromhex(cid): chr(int(char, 16)) for cid, char in pairs}
    encode = {char: cid for cid, char in decode.items()}
    descendant = font['/DescendantFonts'][0].get_object()
    widths = {}
    values = descendant['/W']
    index = 0
    while index < len(values):
        first, second = int(values[index]), values[index + 1]
        index += 2
        if isinstance(second, list):
            widths.update({first + offset: float(value) for offset, value in enumerate(second)})
        else:
            value = float(values[index])
            index += 1
            widths.update({cid: value for cid in range(first, int(second) + 1)})

    def raw_bytes(value):
        return value.original_bytes if hasattr(value, 'original_bytes') else bytes(value)

    def decoded(value):
        raw = raw_bytes(value)
        return ''.join(decode[raw[i:i + 2]] for i in range(0, len(raw), 2))

    def width(text, size):
        return size * 0.05 * sum(widths.get(int.from_bytes(encode[ch], 'big'),
                                          float(descendant.get('/DW', 1000))) for ch in text) / 1000

    replacements = {
        'PinTrace + Execution Tests': ('PinTrace + BigCodeBench', 161, None),
        'Release Currency': ('Release age (exact VA)', 180, None),
        'Security Screening': ('Advisory matching', 180, None),
        'Satisfiability': ('and satisfiability', 161, 699.72),
        'Test Suite Execution': ('BigCodeBench tests', 180, None),
        'Static': ('Static validation', 180, 805.60),
        'Type Check': ('', 180, None),
        'Complete': ('Dependency', 161, None),
        'Advice Set': ('input', 161, None),
        '(2) Advice Extraction (RQ1)': ('(2) Extraction (RQ1 - all PinTrace)', 161, 281.125),
        'Quality Evaluation (RQ2)': ('', 199, None),
        'Environment Validation (RQ3)': ('', 199, None),
    }
    updated = []
    stream = ContentStream(form, writer)
    operations = []
    block = None
    graphical_before = []
    for args, operator in stream.operations:
        if operator == b'BT':
            assert block is None
            block = [(args, operator)]
            continue
        if block is None:
            operations.append((args, operator))
            graphical_before.append((args, operator))
            continue
        block.append((args, operator))
        if operator != b'ET':
            continue
        text = ''.join(decoded(a[0]) for a, op in block if op == b'Tj')
        matrix = next(a for a, op in block if op == b'Tm')
        font_args = next(a for a, op in block if op == b'Tf')
        spec = replacements.get(text.strip())
        if text.strip() == 'Constraint' and float(matrix[4]) > 600:
            spec = ('Release existence', 161, 699.72)
        if text.strip() == '(3)' and abs(float(matrix[5]) - 279.6) < 0.01:
            spec = ('(3) Frozen Dependency Resolution', 199, 492.30)
        if text.strip() == 'Time-aligned Dependency':
            spec = ('', 199, None)
        if text.strip() == 'Resolution' and abs(float(matrix[5]) - 279.6) < 0.01:
            spec = ('', 199, None)
        if text.strip() == '(4)' and abs(float(matrix[5]) - 190.4) < 0.01:
            spec = ('(4) Advice Quality (RQ2 - clean PinTrace)', 161, 763.30)
        if text.strip() == 'Advice' and abs(float(matrix[5]) - 190.4) < 0.01:
            spec = ('', 199, None)
        if text.strip() == '(5)' and abs(float(matrix[5]) - 279.6) < 0.01:
            spec = ('(5) Validation (RQ3 - both datasets)', 180, 763.30)
        if spec is None:
            operations.extend(block)
        else:
            new_text, size, center = spec
            center = center if center is not None else float(matrix[4]) + width(text, float(font_args[1])) / 2
            prefix = []
            for a, op in block:
                a = list(a)
                if op == b'Tf':
                    a[1] = FloatObject(size)
                if op == b'Tm':
                    a[4] = FloatObject(center - width(new_text, size) / 2)
                prefix.append((a, op))
                if op == b'Tm':
                    break
            if new_text:
                prefix.append(([ByteStringObject(b''.join(encode[ch] for ch in new_text))], b'Tj'))
            prefix.append(([], b'ET'))
            operations.extend(prefix)
            updated.append({'old': text, 'new': new_text, 'font_size_pt': size * 0.05})
        block = None
    assert block is None and len(updated) == 19, updated
    graphical_after = []
    in_text = False
    for args, operator in operations:
        if operator == b'BT':
            in_text = True
        elif operator == b'ET':
            in_text = False
        elif not in_text:
            graphical_after.append((args, operator))
    assert graphical_before == graphical_after
    stream.operations = operations
    form.set_data(stream.get_data())
    writer.add_metadata({'/Title': 'Dependency-version advice: study overview'})
    pdf = output / (STEM + '.pdf')
    writer.write(pdf)
    text = PdfReader(pdf).pages[0].extract_text()
    for update in updated:
        if update['new']:
            assert update['new'] in text, update
    assert 'Execution Tests' not in text and 'Release Currency' not in text and 'Advice Set' not in text
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(str(pdf)) as document:
        page = document[0]
        bitmap = page.render(scale=2200 / page.get_width())
        bitmap.to_pil().save(output / (STEM + '.png'))
    obsolete_svg = output / (STEM + '.svg')
    if obsolete_svg.is_file():
        obsolete_svg.unlink()
    report = {'source': str(ORIGINAL.relative_to(ROOT)), 'source_sha256': ORIGINAL_SHA256,
              'updated_labels': updated, 'original_graphics_preserved': True,
              'formats': ['pdf', 'png'],
              'output_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (output / 'fig1_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    parser.add_argument('--poppler-bin', type=Path, default=os.environ.get('POPPLER_BIN'))
    args = parser.parse_args()
    print(json.dumps(build_overview(args.output, args.poppler_bin), ensure_ascii=False))
