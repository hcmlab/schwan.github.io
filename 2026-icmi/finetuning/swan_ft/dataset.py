from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import RunSpec
from .paths import resolver_for
from .prompts import get_prompt, make_human_message, make_response


def load_icep_definitions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    definitions: dict[str, str] = {}
    for role in ["infant", "caregiver"]:
        for code, info in data.get("phases", {}).get(role, {}).items():
            definitions[code.lower()] = info.get("description", info.get("name", ""))
        for code, info in data.get("additional_codes", {}).get(role, {}).items():
            definitions[code.lower()] = info.get("description", info.get("name", ""))
    return definitions


def _find_existing_dataset_files(base_root: Path, dataset_file: str, folds_file: str) -> tuple[Path, Path | None]:
    dataset_path = base_root / dataset_file
    folds_path = base_root / folds_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {dataset_path}")
    if folds_path.exists():
        return dataset_path, folds_path
    return dataset_path, None


def _load_folds_payload(base_root: Path, folds_path: Path | None, folds: list[int]) -> dict[str, Any]:
    if folds_path is not None and folds_path.exists():
        with folds_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("train_dev_sessions", [])
        payload.setdefault("test_sessions", [])
        payload.setdefault("folds", {})
        return payload
    payload: dict[str, Any] = {"train_dev_sessions": [], "test_sessions": [], "folds": {}}
    for fold_id in folds:
        train_path = base_root / f"fold_{fold_id}_train.json"
        val_path = base_root / f"fold_{fold_id}_val.json"
        if not val_path.exists():
            raise FileNotFoundError(
                f"Folds file not found: {base_root / 'folds.json'}. "
                f"Also missing {val_path}. Provide folds.json or fold_{fold_id}_val.json."
            )
        with val_path.open("r", encoding="utf-8") as handle:
            val_samples = json.load(handle)
        val_sessions = sorted({_session_id_from_sample(sample) for sample in val_samples if _session_id_from_sample(sample)})
        payload["folds"][str(fold_id)] = {"val_sessions": val_sessions}
        if train_path.exists():
            with train_path.open("r", encoding="utf-8") as handle:
                train_samples = json.load(handle)
            train_sessions = sorted({_session_id_from_sample(sample) for sample in train_samples if _session_id_from_sample(sample)})
            payload["folds"][str(fold_id)]["train_sessions"] = train_sessions
            payload["train_dev_sessions"] = sorted(set(payload["train_dev_sessions"]).union(train_sessions).union(val_sessions))
    return payload


def _normalize_media_path(base_root: Path, path_value: str) -> Path:
    raw_path = Path(path_value)
    if raw_path.exists():
        return raw_path
    normalized = path_value.replace('\\', '/')
    for marker in ('/clips/', '/audio_clips/', '/videos/'):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            return base_root / marker.strip('/') / Path(suffix)
    if raw_path.is_absolute():
        return raw_path
    return base_root / raw_path


def _parse_response(text: str) -> dict[str, Any]:
    return json.loads(text)


def _effective_role_mode(run_spec: RunSpec) -> str:
    return run_spec.dataset_spec.effective_role_mode(run_spec.options)


def _effective_excluded_labels(run_spec: RunSpec) -> tuple[str, ...]:
    return run_spec.dataset_spec.effective_excluded_labels(run_spec.options)


def _active_allowed_labels(run_spec: RunSpec) -> dict[str, list[str]]:
    prompt = get_prompt(
        role_mode=_effective_role_mode(run_spec),
        include_background=run_spec.dataset_spec.include_background,
        excluded_labels=_effective_excluded_labels(run_spec),
    )
    return prompt["allowed_labels"]


def _sample_allowed(response: dict[str, Any], run_spec: RunSpec) -> bool:
    role_mode = _effective_role_mode(run_spec)
    allowed = _active_allowed_labels(run_spec)
    if role_mode == 'joint':
        return response.get('infant_code') in allowed['infant'] and response.get('caregiver_code') in allowed['caregiver']
    if role_mode == 'infant':
        return response.get('infant_code') in allowed['infant']
    return response.get('caregiver_code') in allowed['caregiver']


def _role_payload(response: dict[str, Any], run_spec: RunSpec, definitions: dict[str, str]) -> dict[str, str]:
    role_mode = _effective_role_mode(run_spec)
    if role_mode == 'infant':
        code = response['infant_code']
        return {
            'infant_code': code,
            'infant_description': response.get('infant_description') or definitions.get(code, ''),
        }
    if role_mode == 'caregiver':
        code = response['caregiver_code']
        return {
            'caregiver_code': code,
            'caregiver_description': response.get('caregiver_description') or definitions.get(code, ''),
        }
    infant_code = response['infant_code']
    caregiver_code = response['caregiver_code']
    return {
        'infant_code': infant_code,
        'caregiver_code': caregiver_code,
        'infant_description': response.get('infant_description') or definitions.get(infant_code, ''),
        'caregiver_description': response.get('caregiver_description') or definitions.get(caregiver_code, ''),
    }


def _extract_audio(video_path: Path, audio_path: Path, overwrite: bool = False) -> None:
    if audio_path.exists() and not overwrite:
        return
    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        raise RuntimeError(
            'ffmpeg is required to create missing audio assets for audio/omni dataset variants. '
            'Install ffmpeg or make sure reusable audio files already exist in the base dataset.'
        )
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        '-y',
        '-i',
        str(video_path),
        '-vn',
        '-acodec',
        'pcm_s16le',
        '-ar',
        '16000',
        '-ac',
        '1',
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _expected_audio_path(base_root: Path, audio_subdir: str, source_video: Path) -> Path:
    rel_parts = source_video.parts[-2:]
    return base_root / audio_subdir / rel_parts[0] / (Path(rel_parts[1]).stem + '.wav')


def _resolve_existing_audio(sample: dict[str, Any], base_root: Path, audio_subdir: str, source_video: Path | None) -> Path | None:
    audios = sample.get('audios') or []
    if audios:
        existing_audio = _normalize_media_path(base_root, audios[0])
        if existing_audio.exists():
            return existing_audio
    if source_video is None:
        return None
    shared_audio = _expected_audio_path(base_root, audio_subdir, source_video)
    if shared_audio.exists():
        return shared_audio
    return None


def _copy_dataset_sample(
    sample: dict[str, Any],
    base_root: Path,
    variant_root: Path,
    run_spec: RunSpec,
    prompt_text: str,
    definitions: dict[str, str],
    overwrite: bool = False,
) -> dict[str, Any] | None:
    response = _parse_response(sample['conversations'][1]['value'])
    if not _sample_allowed(response, run_spec):
        return None

    rebuilt = {
        'conversations': [
            {'from': 'human', 'value': make_human_message(run_spec.dataset_spec.modalities, prompt_text)},
            {'from': 'gpt', 'value': make_response(_role_payload(response, run_spec, definitions))},
        ]
    }

    source_video: Path | None = None
    if 'video' in run_spec.dataset_spec.modalities:
        videos = sample.get('videos') or []
        if not videos:
            raise ValueError('Video modality requested but sample has no videos')
        source_video = _normalize_media_path(base_root, videos[0])
        rebuilt['videos'] = [str(source_video)]

    if 'audio' in run_spec.dataset_spec.modalities:
        if source_video is None:
            videos = sample.get('videos') or []
            if not videos:
                raise ValueError('Audio extraction requires a source video clip')
            source_video = _normalize_media_path(base_root, videos[0])
        if run_spec.dataset_spec.reuse_video_container_for_audio:
            rebuilt['audios'] = [str(source_video)]
            return rebuilt
        existing_audio = _resolve_existing_audio(sample, base_root, run_spec.dataset_spec.audio_subdir, source_video)
        if existing_audio is not None:
            rebuilt['audios'] = [str(existing_audio)]
        else:
            audio_path = variant_root / run_spec.dataset_spec.audio_subdir / source_video.parts[-2] / (source_video.stem + '.wav')
            _extract_audio(source_video, audio_path, overwrite=overwrite)
            rebuilt['audios'] = [str(audio_path)]

    return rebuilt


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_dataset_info(run_spec: RunSpec, folds: list[int], variant_id: str) -> dict[str, Any]:
    columns = {'messages': 'conversations'}
    if 'video' in run_spec.dataset_spec.modalities:
        columns['videos'] = 'videos'
    if 'audio' in run_spec.dataset_spec.modalities:
        columns['audios'] = 'audios'

    info: dict[str, Any] = {
        f'{variant_id}_train_dev': {
            'file_name': 'train_dev.json',
            'formatting': 'sharegpt',
            'columns': columns,
        },
        f'{variant_id}_test': {
            'file_name': 'test.json',
            'formatting': 'sharegpt',
            'columns': columns,
        },
    }
    for fold_id in folds:
        for split in ('train', 'val'):
            info[f'{variant_id}_fold_{fold_id}_{split}'] = {
                'file_name': f'fold_{fold_id}_{split}.json',
                'formatting': 'sharegpt',
                'columns': columns,
            }
    return info


def _collect_validation_sessions(folds_payload: dict[str, Any], fold_id: int) -> set[str]:
    return set(folds_payload['folds'][str(fold_id)]['val_sessions'])


def _collect_train_dev_sessions(folds_payload: dict[str, Any]) -> set[str]:
    explicit = set(folds_payload.get('train_dev_sessions', []))
    if explicit:
        return explicit
    combined: set[str] = set()
    for fold in folds_payload.get('folds', {}).values():
        combined.update(fold.get('train_sessions', []))
        combined.update(fold.get('val_sessions', []))
    return combined


def _collect_test_sessions(folds_payload: dict[str, Any]) -> set[str]:
    return set(folds_payload.get('test_sessions', []))


def _session_id_from_sample(sample: dict[str, Any]) -> str:
    media_paths = sample.get('videos') or sample.get('audios') or []
    if not media_paths:
        return ''
    media_path = Path(str(media_paths[0]).replace('\\', '/'))
    return media_path.parts[-2] if len(media_path.parts) >= 2 else ''


def build_dataset_variant(run_spec: RunSpec, icep_definitions_path: Path | None = None) -> dict[str, Any]:
    resolver = resolver_for(run_spec)
    spec = run_spec.dataset_spec
    base_dataset_path, folds_path = _find_existing_dataset_files(resolver.data_root(), spec.base_dataset_file, spec.base_folds_file)
    with base_dataset_path.open('r', encoding='utf-8') as handle:
        base_samples = json.load(handle)
    folds = list(run_spec.options.folds)
    folds_payload = _load_folds_payload(resolver.data_root(), folds_path, folds)

    role_mode = _effective_role_mode(run_spec)
    excluded_labels = _effective_excluded_labels(run_spec)
    prompt = get_prompt(role_mode=role_mode, include_background=spec.include_background, excluded_labels=excluded_labels)
    prompt_text = str(prompt['text'])
    definitions = load_icep_definitions(icep_definitions_path)
    variant_root = resolver.variant_root(spec, run_spec.options)
    variant_id = resolver.variant_id(spec, run_spec.options)

    rebuilt_samples: list[dict[str, Any]] = []
    for sample in base_samples:
        rebuilt = _copy_dataset_sample(
            sample,
            base_root=resolver.data_root(),
            variant_root=variant_root,
            run_spec=run_spec,
            prompt_text=prompt_text,
            definitions=definitions,
            overwrite=run_spec.options.overwrite,
        )
        if rebuilt is not None:
            rebuilt_samples.append(rebuilt)

    _write_json(resolver.variant_dataset_json(spec, run_spec.options), rebuilt_samples)

    train_dev_sessions = _collect_train_dev_sessions(folds_payload)
    test_sessions = _collect_test_sessions(folds_payload)
    train_dev_samples = [sample for sample in rebuilt_samples if _session_id_from_sample(sample) in train_dev_sessions]
    test_samples = [sample for sample in rebuilt_samples if _session_id_from_sample(sample) in test_sessions]
    _write_json(resolver.variant_train_dev_json(spec, run_spec.options), train_dev_samples)
    _write_json(resolver.variant_test_json(spec, run_spec.options), test_samples)

    fold_counts: dict[str, dict[str, int]] = {}
    for fold_id in folds:
        val_sessions = _collect_validation_sessions(folds_payload, fold_id)
        train_samples = [sample for sample in rebuilt_samples if _session_id_from_sample(sample) in train_dev_sessions and _session_id_from_sample(sample) not in val_sessions]
        val_samples = [sample for sample in rebuilt_samples if _session_id_from_sample(sample) in val_sessions]
        _write_json(resolver.variant_fold_json(spec, fold_id, 'train', run_spec.options), train_samples)
        _write_json(resolver.variant_fold_json(spec, fold_id, 'val', run_spec.options), val_samples)
        fold_counts[str(fold_id)] = {'train': len(train_samples), 'val': len(val_samples)}

    _write_json(resolver.variant_dataset_info(spec, run_spec.options), _build_dataset_info(run_spec, folds, variant_id))
    _write_json(
        resolver.variant_labels_path(spec, run_spec.options),
        {
            'role_mode': role_mode,
            'include_background': spec.include_background,
            'excluded_labels': list(excluded_labels),
            'allowed_labels': prompt['allowed_labels'],
            'schema_fields': prompt['schema_fields'],
        },
    )
    _write_json(
        resolver.variant_manifest_path(spec, run_spec.options),
        {
            'variant_id': variant_id,
            'role_mode': role_mode,
            'modalities': list(spec.modalities),
            'include_background': spec.include_background,
            'excluded_labels': list(excluded_labels),
            'source_dataset': str(base_dataset_path),
            'folds_file': str(folds_path),
            'dataset_json': str(resolver.variant_dataset_json(spec, run_spec.options)),
            'dataset_info': str(resolver.variant_dataset_info(spec, run_spec.options)),
            'train_dev_json': str(resolver.variant_train_dev_json(spec, run_spec.options)),
            'test_json': str(resolver.variant_test_json(spec, run_spec.options)),
            'sample_count': len(rebuilt_samples),
            'train_dev_count': len(train_dev_samples),
            'test_count': len(test_samples),
            'fold_counts': fold_counts,
            'test_sessions': sorted(test_sessions),
        },
    )

    return {
        'variant_root': str(variant_root),
        'variant_id': variant_id,
        'dataset_json': str(resolver.variant_dataset_json(spec, run_spec.options)),
        'dataset_info': str(resolver.variant_dataset_info(spec, run_spec.options)),
        'labels_json': str(resolver.variant_labels_path(spec, run_spec.options)),
        'dataset_manifest': str(resolver.variant_manifest_path(spec, run_spec.options)),
        'train_dev_json': str(resolver.variant_train_dev_json(spec, run_spec.options)),
        'test_json': str(resolver.variant_test_json(spec, run_spec.options)),
        'samples': len(rebuilt_samples),
        'train_dev_samples': len(train_dev_samples),
        'test_samples': len(test_samples),
        'folds': folds,
        'role_mode': role_mode,
        'excluded_labels': list(excluded_labels),
    }


def inspect_dataset_variant(run_spec: RunSpec) -> dict[str, Any]:
    resolver = resolver_for(run_spec)
    spec = run_spec.dataset_spec
    dataset_path = resolver.variant_dataset_json(spec, run_spec.options)
    if not dataset_path.exists():
        dataset_path = resolver.base_dataset_json(spec)
    if not dataset_path.exists():
        return {
            'samples': 0,
            'with_video': 0,
            'with_audio': 0,
            'dataset_path': str(dataset_path),
            'exists': False,
        }
    with dataset_path.open('r', encoding='utf-8') as handle:
        samples = json.load(handle)

    counts = {
        'samples': len(samples),
        'with_video': 0,
        'with_audio': 0,
        'role_mode': _effective_role_mode(run_spec),
        'label_counts': {},
    }
    for sample in samples:
        if sample.get('videos'):
            counts['with_video'] += 1
        if sample.get('audios'):
            counts['with_audio'] += 1
        response = _parse_response(sample['conversations'][1]['value'])
        for key, value in response.items():
            if key.endswith('_code'):
                counts['label_counts'][value] = counts['label_counts'].get(value, 0) + 1

    counts['dataset_path'] = str(dataset_path)
    counts['train_dev_json'] = str(resolver.variant_train_dev_json(spec, run_spec.options))
    counts['test_json'] = str(resolver.variant_test_json(spec, run_spec.options))
    counts['exists'] = True
    return counts
