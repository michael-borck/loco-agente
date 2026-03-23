import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://locoagente.org',
  base: '/docs',
  integrations: [
    starlight({
      title: 'LocoAgente',
      description: 'Can Small Models Think in Loops?',
      favicon: '/robot.svg',
      logo: {
        alt: 'LocoAgente',
        src: './src/assets/robot.svg',
        replacesTitle: false,
      },
      social: [
        { icon: 'external', label: 'Home', href: 'https://locoagente.org' },
        { icon: 'external', label: 'LocoLab', href: 'https://locolabo.org' },
        { icon: 'github', label: 'GitHub', href: 'https://github.com/michael-borck/loco-agente' },
      ],
      customCss: ['./src/styles/custom.css'],
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
          items: [
            { label: 'Future Work', slug: 'future-work' },
          ],
        },
      ],
    }),
  ],
});
