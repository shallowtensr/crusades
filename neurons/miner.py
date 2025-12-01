"""Miner CLI for submitting training code to the tournament."""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

import bittensor as bt

from tournament.chain.manager import ChainManager
from tournament.pipeline.validator import CodeValidator
from tournament.storage.database import get_database
from tournament.storage.models import SubmissionModel


async def submit_code(
    code_path: Path,
    wallet: bt.wallet,
    skip_validation: bool = False,
) -> str | None:
    """Submit training code to the tournament.

    Args:
        code_path: Path to train.py file
        wallet: Bittensor wallet for signing
        skip_validation: Skip local code validation

    Returns:
        Submission ID if successful, None otherwise
    """
    # Read code
    if not code_path.exists():
        print(f"Error: File not found: {code_path}")
        return None

    code = code_path.read_text()

    # Validate code locally
    if not skip_validation:
        validator = CodeValidator()
        result = validator.validate(code)
        if not result.valid:
            print("Code validation failed:")
            for error in result.errors:
                print(f"  - {error}")
            return None
        print("Code validation passed")

    # Calculate code hash
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    print(f"Code hash: {code_hash}")

    # Initialize chain manager
    chain = ChainManager(wallet=wallet)
    await chain.sync_metagraph()

    # Check if miner is registered
    hotkey = wallet.hotkey.ss58_address
    if not chain.is_registered(hotkey):
        print(f"Error: Hotkey {hotkey} is not registered on subnet {chain.netuid}")
        return None

    uid = chain.get_uid_for_hotkey(hotkey)
    print(f"Miner UID: {uid}")

    # For now, store code locally (in production, upload to R2)
    # TODO: Implement R2 upload
    bucket_path = f"submissions/{uid}/{code_hash}/train.py"

    # Create submission in database
    db = await get_database()
    submission = SubmissionModel(
        miner_hotkey=hotkey,
        miner_uid=uid,
        code_hash=code_hash,
        bucket_path=bucket_path,
    )
    await db.save_submission(submission)

    print(f"Submission created: {submission.submission_id}")
    print(f"Status: {submission.status.value}")

    return submission.submission_id


def main():
    parser = argparse.ArgumentParser(description="Submit training code to templar-tournament")
    parser.add_argument(
        "code_path",
        type=Path,
        help="Path to train.py file",
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local code validation",
    )

    args = parser.parse_args()

    # Initialize wallet
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    # Run submission
    submission_id = asyncio.run(
        submit_code(
            code_path=args.code_path,
            wallet=wallet,
            skip_validation=args.skip_validation,
        )
    )

    if submission_id:
        print("\nSubmission successful!")
        print(f"Track status at: /submissions/{submission_id}/status")
        sys.exit(0)
    else:
        print("\nSubmission failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
