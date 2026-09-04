// ESLint flat config.
//
// This replaced .eslintrc.json when the frontend moved to Next 16, which
// REMOVED the `next lint` command entirely. Under Next 16 `next lint` is parsed
// as `next <directory>` and fails with "Invalid project directory provided, no
// such directory: .../lint" — so `npm run lint` now calls eslint directly.
//
// eslint-config-next@16 requires ESLint >= 9 and ships flat-config arrays from
// its subpath exports, so they spread in directly with no FlatCompat shim.

import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypeScript from "eslint-config-next/typescript";

export default [
  // Flat config has no implicit ignores beyond node_modules, so build output
  // and generated files must be named or eslint lints the compiled bundle.
  {
    ignores: [
      ".next/**",
      "out/**",
      "coverage/**",
      "next-env.d.ts",
      "**/*.tsbuildinfo",
    ],
  },
  ...nextCoreWebVitals,
  ...nextTypeScript,
];
