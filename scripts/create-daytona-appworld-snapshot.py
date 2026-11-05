"""Import a published, fixed AppWorld image into the operator's Daytona project."""

from __future__ import annotations

import argparse

from coevobench.runtime.sandbox.daytona import client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="registry image with fixed tag or digest")
    parser.add_argument("--name", required=True, help="new Daytona snapshot name")
    parser.add_argument("--kind", choices=("researcher", "policy"), required=True)
    args = parser.parse_args()
    if args.image.endswith(":latest") or (
        "@sha256:" not in args.image and ":" not in args.image.rsplit("/", 1)[-1]
    ):
        parser.error("use a fixed tag or digest, never latest")
    from daytona import CreateSnapshotParams, Resources

    cpu, memory = (4, 8) if args.kind == "researcher" else (2, 4)
    client().snapshot.create(
        CreateSnapshotParams(
            name=args.name,
            image=args.image,
            resources=Resources(cpu=cpu, memory=memory, disk=10),
        ),
        timeout=1800,
        on_logs=print,
    )
    print("Created:", args.name)


if __name__ == "__main__":
    main()
