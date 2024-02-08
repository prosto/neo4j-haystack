# Contributing

## Environment setup

Fork and clone the repository, then:

```bash
cd scripts
./setup.sh
```

> NOTE:
> If it fails for some reason,
> you'll need to install
> [Hatch](https://hatch.pypa.io/latest/)
> manually.
>
> You can install it with:
>
> ```bash
> python3 -m pip install --user pipx
> pipx install hatch
> ```
>
> Now you can try running `hatch env show` to see
> all available environments and provided scripts.

Hatch automatically manages dependencies for given environments. Default environment can be used for running unit/integration tests.

For local development it is suggested to configure Hatch to create virtual environments inside project's folder. First find the location of the configuration file:

```bash
hatch config find # see where configuration file is located
```

Update configuration with the following:

```toml
[dirs.env]
virtual = ".venv"
```

It is suggested to develop locally with minimal supported python version (see `requires-python` in `pyproject.toml`).
To make sure hatch is creating environments with python version you need set `HATCH_PYTHON` as follows:

```bash
export HATCH_PYTHON=~/.pyenv/versions/3.8.16/bin/python # You might have a different version  
hatch run python --version # Checks python version by creating default hatch environment
```

If you run `hatch env create dev` you should see `.venv/dev` directory created with dependencies you need for development.
You could point your IDE to this particular python environment/interpreter.

## Development

1. create a new branch: `git switch -c feature-or-bugfix-name`
1. edit the code and/or the documentation

**Before committing:**

1. run `hatch run lint:all` for code quality checks and formatting
1. run `hatch run test` to run the tests (fix any issue)
1. if you updated the documentation or the project dependencies:
    1. run `hatch run docs:serve`
    1. go to http://localhost:8000/neo4j-haystack/ and check that everything looks good
1. follow [commit message convention](#commit-message-convention)

Don't bother updating the changelog, we will take care of this.

## Commit message convention

Commit messages must follow the convention based on the
[Angular style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message)
or the [Karma convention](https://karma-runner.github.io/4.0/dev/git-commit-msg.html):

```text
<type>[(scope)]: Subject

[Body]
```

**Subject and body must be valid Markdown.**
Subject must have proper casing (uppercase for first letter
if it makes sense), but no dot at the end, and no punctuation
in general.

Scope and body are optional. Type can be:

- `build`: About packaging, building wheels, etc.
- `chore`: About packaging or repo/files management.
- `ci`: About Continuous Integration.
- `deps`: Dependencies update.
- `docs`: About documentation.
- `feat`: New feature.
- `fix`: Bug fix.
- `perf`: About performance.
- `refactor`: Changes that are not features or bug fixes.
- `style`: A change in code style/format.
- `tests`: About tests.

If you write a body, please add trailers at the end
(for example issues and PR references, or co-authors),
without relying on GitHub's flavored Markdown:

```text
Body.

Issue #10: https://github.com/<namespace>/<project>/issues/10
Related to PR namespace/other-project#15: https://github.com/<namespace>/<other-project>/pull/15
```

These "trailers" must appear at the end of the body,
without any blank lines between them. The trailer title
can contain any character except colons `:`.
We expect a full URI for each trailer, not just GitHub autolinks
(for example, full GitHub URLs for commits and issues,
not the hash or the #issue-number).

We do not enforce a line length on commit messages summary and body,
but please avoid very long summaries, and very long lines in the body,
unless they are part of code blocks that must not be wrapped.

## Pull requests guidelines

Link to any related issue in the Pull Request message.

During the review, we recommend using fixups:

```bash
# SHA is the SHA of the commit you want to fix
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash main
```

And force-push:

```bash
git push -f
```

If this seems all too complicated, you can push or force-push each new commit,
and we will squash them ourselves if needed, before merging.
