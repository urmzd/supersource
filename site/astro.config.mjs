// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import sidebar from './src/generated/sidebar.json' with { type: 'json' };

// Content is generated from the repo's READMEs by scripts/sync-content.mjs.
export default defineConfig({
	site: 'https://supersource.urmzd.com',
	trailingSlash: 'always',
	integrations: [
		starlight({
			title: 'Supersource',
			description: 'A free, runnable curriculum from foundations to Staff-level depth: math, algorithms, ML systems, and infrastructure.',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/urmzd/supersource' }],
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'script',
					attrs: { type: 'module' },
					content: `
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
const dark = document.documentElement.dataset.theme === 'dark';
mermaid.initialize({ startOnLoad: true, theme: dark ? 'dark' : 'neutral' });`,
				},
			],
			sidebar,
		}),
	],
});
