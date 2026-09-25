# Team S3 bucket — original announcement

Posted to the team chat 2026-09-25. Recorded verbatim because it is the only
statement of the bucket's ownership and access model.

> team, shared s3 bucket is ready:
>
> bucket name: amazon-cuda-commandos-2026
> region: ap-south-1 (Mumbai)
>
> you each have full read+write access now. to use it:
>
> 1. install aws cli if you don't have it (brew install awscli on mac)
> 2. run aws configure — enter your own access key id + secret (get from your
>    AWS console → IAM → security credentials → create access key), region:
>    ap-south-1, output format: json
> 3. upload your stuff into your OWN folder so we don't overwrite each other:
>    aws s3 cp ./yourfile.tar.gz s3://amazon-cuda-commandos-2026/yourname/
> 4. pull anyone else's files:
>    aws s3 cp s3://amazon-cuda-commandos-2026/theirname/file.pt ./file.pt
> 5. see what's in the bucket:
>    aws s3 ls s3://amazon-cuda-commandos-2026/
>
> note: this won't show up automatically in your own s3 console — you access it
> directly by name via cli, that's normal for cross-account access

## Deviation from step 2

`docs/TEAM_BUCKET.md` tells people to authenticate with `aws login` instead of
creating a static access key.

Both work. The difference is that `aws configure` with a key pair writes a
long-lived secret to `~/.aws/credentials` in plaintext — it never expires on
its own, has to be rotated by hand, and is the usual way AWS credentials end up
committed to a hackathon repo. `aws login` issues credentials valid 12 hours,
renewable for 90 days, with nothing secret written to disk.

Verified working on account `654479364872` with the `amlc` profile:
`aws s3 ls s3://amazon-cuda-commandos-2026/` returns successfully (bucket
currently empty).

The note about the bucket not appearing in your own S3 console is correct —
cross-account grants are addressed by name over the CLI.
