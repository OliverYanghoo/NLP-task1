import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def load_method_metrics(results_dir: Path):
    rows = []
    for json_path in sorted(results_dir.glob('*.json')):
        if json_path.name in {'split.json', 'dataset.json'}:
            continue
        try:
            with json_path.open('r', encoding='utf-8') as stream:
                payload = json.load(stream)
        except Exception:
            continue
        metrics = payload.get('test', {})
        rows.append({
            'method': payload.get('method', json_path.stem),
            'accuracy': float(metrics.get('accuracy', 0.0)),
            'macro_f1': float(metrics.get('macro_f1', 0.0)),
            'best_C': payload.get('best_C'),
        })
    return rows


def generate_report(results_dir: Path, output_path: Path):
    rows = load_method_metrics(results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, leading=22, spaceAfter=18)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=12, leading=16, spaceAfter=12)

    story = [Paragraph('Text Classification Experiment Summary', title_style)]
    story.append(Paragraph(f'Generated from: {results_dir}', styles['BodyText']))
    story.append(Spacer(1, 0.3 * inch))

    if not rows:
        story.append(Paragraph('No experiment result files were found.', styles['BodyText']))
        doc.build(story)
        return output_path

    data = [['Method', 'Accuracy', 'Macro-F1', 'Best C']]
    for row in rows:
        data.append([
            row['method'],
            f"{row['accuracy']:.4f}",
            f"{row['macro_f1']:.4f}",
            '-' if row['best_C'] is None else str(row['best_C']),
        ])

    table = Table(data, colWidths=[2.0 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4C78A8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph('Summary', section_style))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph('Notes: All metrics are evaluated on the test split after selecting the best validation configuration.', styles['BodyText']))

    doc.build(story)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate a PDF report from saved experiment metrics.')
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/summary_report.pdf'))
    args = parser.parse_args()

    generate_report(args.results_dir, args.output)
    print(f'PDF report saved to {args.output}')


if __name__ == '__main__':
    main()
