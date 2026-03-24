#!/usr/bin/env python3
"""
reorg_plots.py

Reorganize plot files under data/ (e.g. data/512, data/1024) into type-based subfolders.

Features:
- Classify files by regex rules and extension-based rules
- Supports modes: move, copy, symlink
- Dry-run by default; use --apply to perform operations
- Writes a manifest JSON (reorg_manifest_<timestamp>.json) recording planned/applied actions
- Optional: generate a simple index.html per resolution for browsing
- Rollback using manifest: --rollback <manifest.json>

Examples:
  python scripts/reorg_plots.py --root data --mode move         # dry-run (no changes)
  python scripts/reorg_plots.py --root data --mode move --apply --manifest manifest.json
  python scripts/reorg_plots.py --rollback manifest.json

"""

import argparse
import json
import logging
import os
import shutil
import sys
import re
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Default classification rules (ordered)
DEFAULT_MAPPING = [
    ("spectrum", re.compile(r'(?:\b(?:energy|kinetic|generalized|potential|pressure|enstrophy)_spectr(?:um)?\b|spectrum)', re.I)),
    ("time-series", re.compile(r'(_vs_t|P2_vs_t|Qbox_vs_t|energy_vs_t|reynolds_power_vs_t|zonal_energy_fraction_vs_t|enstrophy_vs_t|cum_reynolds_power_vs_t|dQdt_vs_t|DH_vs_t|Dchi_vs_t|T1_vs_t|T2_vs_t|T3_vs_t)', re.I)),
    ("flux", re.compile(r'(?:\bE_flux\b|\bG_flux\b|energy_flux|_flux_)', re.I)),
    ("dissipation", re.compile(r'dissipation', re.I)),
    ("injection", re.compile(r'injection|energy_injection|E_injection', re.I)),
    ("kxky", re.compile(r'(?:kx[_-]?ky|E_kx_ky|kxky)', re.I)),
    ("xt-maps", re.compile(r'(_xt_|R_and_P_xt|RP_xt|RPhi_xt|PP_xt|PPhi_xt|P_xt|Q_xt|_xt_plots)', re.I)),
    ("vbar", re.compile(r'\bvbar\b|vbar_xt|vbar_R_xt', re.I)),
    ("zonal", re.compile(r'zonal|zonal_energy', re.I)),
]

EXT_CATEGORIES = {
    '.mp4': 'video',
    '.h5': 'data',
    '.hdf5': 'data',
}

FALLBACK_CATEGORY = 'reynolds-stress'


def detect_resolutions(root: Path, provided: str | None):
    if provided:
        return [s.strip() for s in provided.split(',') if s.strip()]
    # default: treat immediate numeric subdirectories as resolutions (e.g., 512, 1024)
    res = [p.name for p in root.iterdir() if p.is_dir()]
    return res


def classify_file(path: Path):
    ext = path.suffix.lower()
    if ext in EXT_CATEGORIES:
        return EXT_CATEGORIES[ext]
    name = path.name
    for cat, regex in DEFAULT_MAPPING:
        if regex.search(name):
            return cat
    # additional heuristics
    if ext in ('.png', '.jpg', '.jpeg', '.svg'):
        # try to detect energy/spectrum by name
        if re.search(r'energy|spectrum|kappa|kinetic|enstrophy', name, re.I):
            return 'spectrum'
    if ext == '.pdf':
        if re.search(r'spectrum|spectrum_kapt|energy_spectrum', name, re.I):
            return 'spectrum'
    return FALLBACK_CATEGORY


def unique_dest(dst: Path) -> Path:
    if not dst.exists():
        return dst
    parent = dst.parent
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def plan_moves(root: Path, resolutions, recursive: bool):
    moves = []
    for res in resolutions:
        res_dir = root / res
        if not res_dir.exists() or not res_dir.is_dir():
            logging.warning("Resolution folder not found, skipping: %s", res_dir)
            continue
        if recursive:
            files = [p for p in res_dir.rglob('*') if p.is_file() and p.parent != res_dir]
            # The above excludes files deeper than 1 level if parent == res_dir is false — change to include all files
            files = [p for p in res_dir.rglob('*') if p.is_file()]
        else:
            files = [p for p in res_dir.iterdir() if p.is_file()]
        for p in files:
            # skip files already in a subfolder that looks like a category: skip if parent is not the resolution dir
            if p.parent != res_dir:
                # still allow if recursive is True (we will reorganize nested files too)
                if not recursive:
                    continue
            cat = classify_file(p)
            dst_dir = res_dir / cat
            dst = dst_dir / p.name
            dst = unique_dest(dst)
            moves.append({'src': str(p), 'dst': str(dst), 'category': cat})
    return moves


def apply_moves(moves, mode='move', apply=False, manifest_path=None):
    manifest = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'mode': mode,
        'apply': bool(apply),
        'moves': []
    }
    for m in moves:
        src = Path(m['src'])
        dst = Path(m['dst'])
        entry = {'src': str(src), 'dst': str(dst), 'category': m.get('category'), 'action': mode, 'status': 'planned'}
        manifest['moves'].append(entry)

    if manifest_path:
        try:
            Path(manifest_path).write_text(json.dumps(manifest, indent=2))
            logging.info('Wrote manifest (planned) to %s', manifest_path)
        except Exception as e:
            logging.warning('Could not write manifest: %s', e)

    if not apply:
        # dry run: summarize
        logging.info('\nDry-run: %d files would be organized (no changes made).', len(moves))
        cats = {}
        for m in moves:
            cats.setdefault(m['category'], 0)
            cats[m['category']] += 1
        for k, v in sorted(cats.items()):
            logging.info('  %s: %d', k, v)
        return manifest

    # perform actions
    for entry in manifest['moves']:
        src = Path(entry['src'])
        dst = Path(entry['dst'])
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if entry['action'] == 'move':
                # use shutil.move
                if src.exists():
                    shutil.move(str(src), str(dst))
                    entry['status'] = 'moved'
                else:
                    entry['status'] = 'skipped_missing_src'
            elif entry['action'] == 'copy':
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                    entry['status'] = 'copied'
                else:
                    entry['status'] = 'skipped_missing_src'
            elif entry['action'] == 'symlink':
                # create a relative symlink from dst -> src
                if not dst.exists():
                    rel = os.path.relpath(str(src), str(dst.parent))
                    os.symlink(rel, str(dst))
                    entry['status'] = 'symlinked'
                else:
                    entry['status'] = 'skipped_exists'
            else:
                entry['status'] = 'unknown_action'
        except Exception as e:
            entry['status'] = 'error'
            entry['error'] = str(e)
    # write manifest with results
    if manifest_path:
        try:
            Path(manifest_path).write_text(json.dumps(manifest, indent=2))
            logging.info('Wrote manifest (applied) to %s', manifest_path)
        except Exception as e:
            logging.warning('Could not write manifest after applying: %s', e)
    return manifest


def rollback(manifest_path: Path):
    if not manifest_path.exists():
        logging.error('Manifest not found: %s', manifest_path)
        return
    data = json.loads(manifest_path.read_text())
    moves = data.get('moves', [])
    # reverse operations
    for entry in reversed(moves):
        src = Path(entry['src'])
        dst = Path(entry['dst'])
        action = entry.get('action')
        try:
            if action == 'move':
                # if dst exists, move it back to src (avoid overwriting)
                if dst.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    target = src
                    if src.exists():
                        target = unique_dest(src)
                    shutil.move(str(dst), str(target))
                    logging.info('Restored %s -> %s', dst, target)
                else:
                    logging.debug('Destination missing for move rollback: %s', dst)
            elif action == 'copy':
                if dst.exists():
                    if dst.is_file() or dst.is_symlink():
                        dst.unlink()
                        logging.info('Removed copied file %s', dst)
            elif action == 'symlink':
                if dst.exists() and dst.is_symlink():
                    dst.unlink()
                    logging.info('Removed symlink %s', dst)
        except Exception as e:
            logging.warning('Rollback failed for %s: %s', dst, e)


def generate_index(root: Path, resolutions):
    # create a simple index.html under each resolution listing categories and files
    for res in resolutions:
        res_dir = root / res
        if not res_dir.exists():
            continue
        categories = [p for p in res_dir.iterdir() if p.is_dir()]
        lines = [
            '<!doctype html>',
            '<html><head><meta charset="utf-8"><title>Index - %s</title>' % res,
            '<style>body{font-family:Arial} img{max-width:240px;height:auto;margin:6px;border:1px solid #ccc}</style>',
            '</head><body>',
            '<h1>Plots: %s</h1>' % res,
        ]
        for c in sorted(categories, key=lambda p: p.name):
            lines.append('<h2>%s</h2>' % c.name)
            lines.append('<div>')
            for f in sorted(c.iterdir()):
                rel = os.path.relpath(str(f), str(res_dir))
                ext = f.suffix.lower()
                if ext in ('.png', '.jpg', '.jpeg', '.svg'):
                    lines.append('<a href="%s"><img src="%s" alt="%s"></a>' % (rel, rel, f.name))
                elif ext == '.mp4':
                    lines.append('<div><video width="360" controls src="%s"></video><div>%s</div></div>' % (rel, f.name))
                else:
                    lines.append('<div><a href="%s">%s</a></div>' % (rel, f.name))
            lines.append('</div>')
        lines.append('</body></html>')
        idx_path = res_dir / 'index.html'
        idx_path.write_text('\n'.join(lines))
        logging.info('Generated index: %s', idx_path)


def parse_args():
    p = argparse.ArgumentParser(description='Reorganize plot files into type-based subfolders (dry-run default).')
    p.add_argument('--root', default='data', help='Root data directory (default: data)')
    p.add_argument('--resolutions', help='Comma-separated resolution folders to process (default: autodetect)')
    p.add_argument('--mode', choices=['move', 'copy', 'symlink'], default='move', help='Action to use when reorganizing (default: move)')
    p.add_argument('--apply', action='store_true', help='Perform actions. Without this flag the script runs a dry-run.')
    p.add_argument('--manifest', help='Path to write manifest JSON (default: reorg_manifest_<timestamp>.json).')
    p.add_argument('--generate-index', action='store_true', help='Generate simple index.html files under each resolution.')
    p.add_argument('--recursive', action='store_true', help='Recurse into subdirectories (use with care).')
    p.add_argument('--rollback', help='Rollback changes using manifest JSON')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if args.rollback:
        rollback(Path(args.rollback))
        return
    if not root.exists() or not root.is_dir():
        logging.error('Root data directory not found: %s', root)
        sys.exit(2)
    resolutions = detect_resolutions(root, args.resolutions)
    if not resolutions:
        logging.error('No resolution directories found under %s', root)
        sys.exit(2)
    logging.info('Detected resolutions: %s', ', '.join(resolutions))
    moves = plan_moves(root, resolutions, recursive=args.recursive)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    manifest_path = args.manifest or (Path.cwd() / f'reorg_manifest_{timestamp}.json')
    manifest = apply_moves(moves, mode=args.mode, apply=args.apply, manifest_path=str(manifest_path))
    if args.generate_index and args.apply:
        generate_index(root, resolutions)
    # final summary
    if not args.apply:
        logging.info('\nDry-run complete. Use --apply to execute. Example:')
        logging.info('  python %s --root %s --mode %s --apply --manifest %s', Path(__file__).name, args.root, args.mode, manifest_path)
    else:
        logging.info('\nApply complete. Manifest: %s', manifest_path)
        logging.info('To rollback use: python %s --rollback %s', Path(__file__).name, manifest_path)

if __name__ == '__main__':
    main()
