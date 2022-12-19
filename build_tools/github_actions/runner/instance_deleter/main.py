# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Cloud Functions proxy enabling GCE VMs in a Managed Instance Group to delete themselves.

GCE Managed instance groups don't have any good way to handle autoscaling for
long-running workloads. With the autoscaler configured to scale in, instances
get only 90 seconds warning to shut down. So we set the autoscaler to only scale
out and have the VMs tear themselves down when they're down with their work.
This is the approach suggested by the managed instance group team:

https://drive.google.com/file/d/1XlwxF_0T7pUnbzhL5ePDoW-Q3GAaLO11

But anything that brings down the VM other than a delete call to the instance
group manager API makes the VM get considered "unhealthy", which means it gets
recreated in exactly the same configuration, regardless of any update or
autoscaling settings. Making the correct API call requires broad permissions on
the instance group manager, which we don't want to give the VMs. To scope
permissions to individual instances, this proxy service makes use of instance
identity tokens to allow an instance to make a call only to delete itself.

See
https://cloud.google.com/compute/docs/instances/verifying-instance-identity

This makes use of the GCP Cloud Functions serverless offering. It's another
level of abstraction on top of Cloud Run, where you don't even need to create your
own docker container. For local development:

  functions-framework --target=delete_self
  curl -X DELETE -v --header "Authorization: Bearer $(cat /tmp/token.txt)" localhost:8080

You'll need to get a token that corresponds to an actual instance though or
you'll get an error:

  gcloud compute ssh github-runner-testing-presubmit-cpu-us-west1-h58j \
    --user-output-enabled=false \
    --command "curl -sSfL \
        -H 'Metadata-Flavor: Google' \
        'http://metadata/computeMetadata/v1/instance/service-accounts/default/identity?audience=localhost&format=full'" \
    > /tmp/token.txt

To deploy:
  gcloud functions deploy instance-self-deleter \
    --gen2 \
    --runtime=python310 \
    --region=us-central1 \
    --source=. \
    --entry-point=delete_self \
    --trigger-http

See https://cloud.google.com/functions/docs for more details.
"""

import flask
import functions_framework
import google.api_core.exceptions
import google.auth.exceptions
import requests
from google.auth import transport
from google.cloud import compute
from google.oauth2 import id_token

AUTH_HEADER_PREFIX = "Bearer "
MIG_METADATA_KEY = "created-by"

instances_client = compute.InstancesClient()
migs_client = compute.RegionInstanceGroupManagersClient()
session = requests.Session()

print("Server started")


def _verify_token(token: str) -> dict:
  """Verify token signature and return the token payload"""
  request = transport.requests.Request(session)
  payload = id_token.verify_oauth2_token(token, request=request)
  return payload


def _get_region(zone: str) -> str:
  """Extract region name from zone name"""
  # Drop the trailing zone identifier to get the region. Yeah it kinda does seem
  # like there should be a better way to do this...
  region, _ = zone.rsplit("-", maxsplit=1)
  return region


def _get_name_from_resource(resource: str) -> str:
  """Extract just the final name component from a fully scoped resource name."""
  _, name = resource.rsplit("/", maxsplit=1)
  return name


def _get_from_items(items: compute.Items, key: str):
  # Why would the GCP Python API return something as silly as a dictionary?
  return next((item.value for item in items if item.key == key), None)


@functions_framework.http
def delete_self(request):
  """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    Note:
        For more information on how Flask integrates with Cloud
        Functions, see the `Writing HTTP functions` page.
        <https://cloud.google.com/functions/docs/writing/http#http_frameworks>
  """
  if request.method != "DELETE":
    return flask.abort(
        400, f"Invalid method {request.method}. Only DELETE is supported.")

  # No path is needed, since the token contains all the information we need.
  if request.path != "/":
    return flask.abort(
        400, f"Invalid request path {request.path}. Only root path is valid).")

  auth_header = request.headers.get("Authorization")
  if auth_header is None:
    return flask.abort(401, "Authorization header is missing")
  if not auth_header.startswith(AUTH_HEADER_PREFIX):
    return flask.abort(
        401, f"Authorization header does not start with expected string"
        f" {AUTH_HEADER_PREFIX}.")

  token = auth_header[len(AUTH_HEADER_PREFIX):]

  try:
    # We don't verify audience here because Cloud IAM will have already done so
    # and jwt's matching of audiences is exact, which means trailing slashes or
    # http vs https matters and that's pretty brittle.
    token_payload = _verify_token(token)
  except (ValueError, google.auth.exceptions.GoogleAuthError) as e:
    print(e)
    return flask.abort(401, "Decoding bearer token failed.")

  print(f"Token payload: {token_payload}")

  try:
    compute_info = token_payload["google"]["compute_engine"]
  except KeyError:
    return flask.abort(
        401, "Bearer token payload does not have expected field google.compute")

  project = compute_info["project_id"]
  zone = compute_info["zone"]
  instance_name = compute_info["instance_name"]
  try:
    instance = instances_client.get(instance=instance_name,
                                    project=project,
                                    zone=zone)
  except google.api_core.exceptions.NotFound as e:
    print(e)
    return flask.abort(
        404,
        f"Instance {instance_name} not found for zone={zone}, project={project}"
    )

  instance_id = int(compute_info["instance_id"])
  # Verify it's *actually* the same instance. Names get reused, but IDs
  # don't. For some reason you can't reference instances by their ID in any
  # of the APIs.
  if instance.id != instance_id:
    return flask.abort(
        400,
        f"Existing instance of the same name {instance.name} has a different"
        f" ID {instance.id} than token specifies {instance_id}.")

  mig = _get_from_items(instance.metadata.items, MIG_METADATA_KEY)

  if mig is None:
    return flask.abort(400,
                       (f"Instance is not part of a managed instance group."
                        f" Did not find {MIG_METADATA_KEY} in metadata."))
  mig = _get_name_from_resource(mig)

  try:
    operation = migs_client.delete_instances(
        instance_group_manager=mig,
        project=project,
        region=_get_region(zone),
        # For some reason we can't just use a list of instance names and need to
        # build this RhymingRythmicJavaClasses proto. Also, unlike all the other
        # parameters, the instance has to be a fully-specified URL for the
        # instance, not just its name.
        region_instance_group_managers_delete_instances_request_resource=(
            compute.RegionInstanceGroupManagersDeleteInstancesRequest(
                instances=[instance.self_link])))
  except Exception as e:
    # An error here means we screwed up the syntax of the request. That is
    # almost certainly a server error
    return flask.abort(500,
                       f"Error requesting that {mig} delete {instance_name}.")

  try:
    # This is actually an extended operation that you have to poll to get its
    # status, but we just check the status once because it appears that errors
    # always show up here.
    operation.result()
  except google.api_core.exceptions.ClientError as e:
    print(e)
    # Unpack the actual usable error message
    msg = f"Error requesting that {mig} delete {instance_name}:" "\n" + "\n".join(
        [f"{err.code}: {err.message}" for err in e.response.error.errors])
    print(msg)
    # We're not actually totally sure whether this is a client or server error
    # for the overall request, but let's call it a client error (the only client
    # here is our VM instances, so I think we can be a bit loose).
    return flask.abort(400, msg)

  return f"{instance_name} has been marked for deletion by {mig}."
