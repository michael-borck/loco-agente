import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { loco } from '@locolabo/theme';

export default defineConfig({
  site: 'https://locoagente.org',
  integrations: [
    starlight(
      loco({
        project: 'loco-agente',
        title: 'LocoAgente',
        description: 'Can Small Models Think in Loops?',
        home: 'https://locoagente.org',
        github: 'loco-agente',
        accent: { low: '#021a1e', main: '#06b6d4', high: '#a5f3fc' },
        logo: {
          alt: 'LocoAgente',
          src: './src/assets/art/icon.png',
          replacesTitle: false,
        },
        sidebar: [
          {
            label: 'Background',
            items: [
              { label: 'Why LocoAgente', slug: 'why-locoagente' },
              { label: 'What Are Agents?', slug: 'what-are-agents' },
              { label: 'Agentic Drift', slug: 'agentic-drift' },
              { label: 'Scaffolding Strategies', slug: 'scaffolding-strategies' },
            ],
          },
          {
            label: 'Research',
            items: [
              { label: 'Research Plan', slug: 'research-plan' },
              { label: 'Track A: Autoresearch', slug: 'track-a-autoresearch' },
              { label: 'Track B: Task Agents', slug: 'track-b-task-agents' },
              { label: 'Track C: Scaffolding', slug: 'track-c-scaffolding' },
              { label: 'Track D: Framework Eval', slug: 'track-d-framework-eval' },
            ],
          },
          {
            label: 'Reference',
            items: [{ label: 'Future Work', slug: 'future-work' }],
          },
        ],
      })
    ),
  ],
});
