#!/usr/bin/env python3
# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Upload benchmark results to IREE Benchmark Dashboards.

This script is meant to be used by Buildkite for automation.

Example usage:
  # Export necessary environment variables:
  export IREE_DASHBOARD_URL=...
  export IREE_DASHBOARD_API_TOKEN=...
  # Then run the script:
  python3 upload_benchmarks.py /path/to/benchmark/json/file
"""

import argparse
import json
import os
import re
import requests
import subprocess
import time

from typing import Any, Dict, Optional

from common.benchmark_description import (BenchmarkInfo, BenchmarkResults,
                                          get_output)

IREE_GITHUB_COMMIT_URL_PREFIX = 'https://github.com/google/iree/commit'
IREE_PROJECT_ID = 'IREE'
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# A non-exhaustive list of models and their source URLs.
# For models listed here we can provide a nicer description for them on
# webpage.
IREE_TF_MODEL_SOURCE_URL = {
    'MobileNetV2':
        'https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2',
    'MobileNetV3Small':
        'https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small',
    'MobileBertSquad':
        'https://github.com/google-research/google-research/tree/master/mobilebert',
}


def get_model_description(benchmark_info: BenchmarkInfo) -> Optional[str]:
  """Gets the model description for the given benchmark."""
  url = None
  name = benchmark_info.model_name
  if benchmark_info.model_source == "TensorFlow":
    url = IREE_TF_MODEL_SOURCE_URL.get(name)
  if url is not None:
    description = f'{name} from <a href="{url}">{url}</a>.'
    return description
  return None


def get_git_commit_hash(commit: str, verbose: bool = False) -> str:
  """Gets the commit hash for the given commit."""
  return get_output(['git', 'rev-parse', commit],
                    cwd=THIS_DIRECTORY,
                    verbose=verbose)


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
  """Gets the total commit count in history ending with the given commit."""
  count = get_output(['git', 'rev-list', '--count', commit],
                     cwd=THIS_DIRECTORY,
                     verbose=verbose)
  return int(count)


def get_git_commit_info(commit: str, verbose: bool = False) -> Dict[str, str]:
  """Gets commit information dictory for the given commit."""
  cmd = [
      'git', 'show', '--format=%H:::%h:::%an:::%ae:::%s', '--no-patch', commit
  ]
  info = get_output(cmd, cwd=THIS_DIRECTORY, verbose=verbose)
  segments = info.split(':::')
  return {
      'hash': segments[0],
      'abbrevHash': segments[1],
      'authorName': segments[2],
      'authorEmail': segments[3],
      'subject': segments[4],
  }


def compose_series_payload(project_id: str,
                           series_id: str,
                           series_description: bool = None,
                           average_range: str = '5%',
                           average_min_count: int = 3,
                           better_criterion: str = 'smaller',
                           override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a series."""
  payload = {
      'projectId': project_id,
      'serieId': series_id,
      'analyse': {
          'benchmark': {
              'range': average_range,
              'required': average_min_count,
              'trend': better_criterion,
          }
      },
      'override': override,
  }
  if series_description is not None:
    payload['description'] = series_description
  return payload


def compose_build_payload(project_id: str,
                          project_github_comit_url: str,
                          build_id: int,
                          commit: str,
                          override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a build."""
  commit_info = get_git_commit_info(commit)
  commit_info['url'] = f'{project_github_comit_url}/{commit_info["hash"]}'
  return {
      'projectId': project_id,
      'build': {
          'buildId': build_id,
          'infos': commit_info,
      },
      'override': override,
  }


def compose_sample_payload(project_id: str,
                           series_id: str,
                           build_id: int,
                           sample_value: int,
                           override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a sample."""
  return {
      'projectId': project_id,
      'serieId': series_id,
      'sample': {
          'buildId': build_id,
          'value': sample_value
      },
      'override': override
  }


def get_required_env_var(var: str) -> str:
  """Gets the value for a required environment variable."""
  value = os.getenv(var)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def post_to_dashboard(url: str,
                      payload: Dict[str, Any],
                      dry_run: bool = False,
                      verbose: bool = False):
  data = json.dumps(payload)

  if dry_run or verbose:
    print(f'API request payload: {data}')

  if dry_run:
    return

  api_token = get_required_env_var('IREE_DASHBOARD_API_TOKEN')
  headers = {
      'Content-type': 'application/json',
      'Authorization': f'Bearer {api_token}',
  }

  response = requests.post(url, data=data, headers=headers)
  code = response.status_code
  if code != 200:
    raise requests.RequestException(
        f'Failed to post to dashboard server with status code {code}')


def add_new_iree_series(series_id: str,
                        series_description: Optional[str] = None,
                        override: bool = False,
                        dry_run: bool = False,
                        verbose: bool = False):
  """Posts a new series to the dashboard."""
  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = compose_series_payload(IREE_PROJECT_ID,
                                   series_id,
                                   series_description,
                                   override=override)
  post_to_dashboard(f'{url}/apis/addSerie',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def add_new_iree_build(build_id: int,
                       commit: str,
                       override: bool = False,
                       dry_run: bool = False,
                       verbose: bool = False):
  """Posts a new build to the dashboard."""
  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = compose_build_payload(IREE_PROJECT_ID,
                                  IREE_GITHUB_COMMIT_URL_PREFIX, build_id,
                                  commit, override)
  post_to_dashboard(f'{url}/apis/addBuild',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def add_new_sample(series_id: str,
                   build_id: int,
                   sample_value: int,
                   override: bool = False,
                   dry_run: bool = False,
                   verbose: bool = False):
  """Posts a new sample to the dashboard."""
  url = get_required_env_var('IREE_DASHBOARD_URL')
  payload = compose_sample_payload(IREE_PROJECT_ID, series_id, build_id,
                                   sample_value, override)
  post_to_dashboard(f'{url}/apis/addSample',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument('benchmark_files',
                      metavar='<benchmark-json-file>',
                      type=check_file_path,
                      nargs='+',
                      help='Path to the JSON file containing benchmark results')
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Print the comment instead of posting to dashboard")
  parser.add_argument('--verbose',
                      action='store_true',
                      help='Print internal information during execution')
  args = parser.parse_args()

  return args


def main(args):
  # Collect benchmark results from all files.
  all_results = []
  for benchmark_file in args.benchmark_files:
    with open(benchmark_file) as f:
      content = f.read()
    all_results.append(BenchmarkResults.from_json_str(content))
  for other_results in all_results[1:]:
    all_results[0].merge(other_results)
  all_results = all_results[0]

  # Register a new build for the current commit.
  commit_hash = get_git_commit_hash(all_results.commit, verbose=args.verbose)
  commit_count = get_git_total_commit_count(commit_hash, verbose=args.verbose)
  add_new_iree_build(commit_count,
                     commit_hash,
                     dry_run=args.dry_run,
                     verbose=args.verbose)

  # Get the mean time for all benchmarks.
  aggregate_results = {}
  for benchmark_index in range(len(all_results.benchmarks)):
    benchmark_case = all_results.benchmarks[benchmark_index]
    benchmark_info = benchmark_case["benchmark"]

    # Make sure each benchmark has a unique name.
    name = str(benchmark_info)
    if name in aggregate_results:
      raise ValueError(f"Duplicated benchmarks: {name}")

    mean_time = all_results.get_aggregate_time(benchmark_index, "mean")
    aggregate_results[name] = (mean_time, benchmark_info)

  # Upload benchmark results to the dashboard.
  for series_id, (sample_value, benchmark_info) in aggregate_results.items():
    model_description = get_model_description(benchmark_info)

    # Override by default to allow updates to the series.
    add_new_iree_series(series_id,
                        model_description,
                        override=True,
                        dry_run=args.dry_run,
                        verbose=args.verbose)
    add_new_sample(series_id,
                   commit_count,
                   sample_value,
                   dry_run=args.dry_run,
                   verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
