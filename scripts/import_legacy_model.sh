#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${script_dir}/.." && pwd)"
source_weight="${repository_dir}/../acne-CV/app/model/yolo_acne_detection.weights.h5"
destination_dir="${repository_dir}/models/legacy/acne-cv"
destination_weight="${destination_dir}/yolo_acne_detection.weights.h5"
expected_size="47016328"
expected_sha256="0aa53871402aa540ebebd6368151eb2d21cf4fa9fa1a41b742ac3facc18f0f35"

if [[ ! -f "${source_weight}" ]]; then
  echo "기존 가중치를 찾지 못했습니다: ${source_weight}" >&2
  exit 1
fi

actual_size="$(stat -f '%z' "${source_weight}")"
actual_sha256="$(shasum -a 256 "${source_weight}" | awk '{print $1}')"

if [[ "${actual_size}" != "${expected_size}" ]]; then
  echo "기존 가중치의 크기가 기록과 다릅니다: ${actual_size}" >&2
  exit 1
fi

if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "기존 가중치의 SHA-256이 기록과 다릅니다: ${actual_sha256}" >&2
  exit 1
fi

mkdir -p "${destination_dir}"

if [[ -f "${destination_weight}" ]]; then
  destination_sha256="$(shasum -a 256 "${destination_weight}" | awk '{print $1}')"
  if [[ "${destination_sha256}" == "${expected_sha256}" ]]; then
    echo "검증된 가중치가 이미 있습니다: ${destination_weight}"
    exit 0
  fi

  echo "대상 경로에 다른 파일이 있습니다. 기존 파일을 덮어쓰지 않습니다: ${destination_weight}" >&2
  exit 1
fi

cp "${source_weight}" "${destination_weight}"
echo "가중치를 복사하고 SHA-256을 확인했습니다: ${destination_weight}"
