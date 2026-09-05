# Coding Standards

Read by `/code-review`'s Standards axis, on top of the Fowler smell baseline it always applies.

## Lints

Clippy `pedantic` and `nursery` groups are denied (see `Cargo.toml` `[lints.clippy]`), plus explicit denies on `unwrap`, `expect`, `panic`, `todo`, `unimplemented`, `unreachable`, indexing/slicing, arithmetic side effects, `as` conversions, and process `exit`. Don't write code that trips these; don't add `#[allow(...)]` to route around them without a comment explaining why.

Tests are exempt: `unwrap`, `expect`, `panic`, and indexing/slicing are allowed in `#[cfg(test)]` code (`clippy.toml`).

## Formatting

`rustfmt` defaults. No manual formatting overrides.

## Enforcement

Both run as pre-commit hooks (`devenv.nix` → `git-hooks.hooks`). A commit that fails either doesn't land.
