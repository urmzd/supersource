# Supersource site

The curriculum as a website, built with [Starlight](https://starlight.astro.build) and deployed to GitHub Pages by `.github/workflows/pages.yml`.

The repo's markdown is the source of truth. `scripts/sync-content.mjs` copies it into `src/content/docs/` at build time, rewrites links, and generates the sidebar. Only `src/content/docs/index.mdx` (the home page) is authored here.

```bash
pnpm install
pnpm dev      # sync + dev server at http://localhost:4321/
pnpm build    # sync + static build into dist/
```
