import argparse
from dataclasses import replace
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from acne_care.api.app import create_app
from acne_care.api.settings import Settings


PRIVATE_NETWORKS = tuple(IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def preview_address(value: str) -> str:
    try:
        address = IPv4Address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter the Mac's private Wi-Fi IPv4 address") from error
    if not any(address in network for network in PRIVATE_NETWORKS):
        raise argparse.ArgumentTypeError("enter the Mac's private Wi-Fi IPv4 address")
    return str(address)


def main():
    parser = argparse.ArgumentParser(description="Run the local acne-care photo analysis web app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mobile-preview", metavar="LAN_IP", type=preview_address,
                        help="preview on the same Wi-Fi with temporary photo storage")
    args = parser.parse_args()
    if args.mobile_preview:
        with TemporaryDirectory(prefix="acne-care-preview-") as temporary:
            settings = replace(Settings.from_environment(), storage_dir=Path(temporary), database_path=None)
            print(f"휴대폰 미리보기 http://{args.mobile_preview}:{args.port}/", flush=True)
            uvicorn.run(create_app(settings, preview_address=args.mobile_preview), host=args.mobile_preview,
                        port=args.port, access_log=False, proxy_headers=False)
        return
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
