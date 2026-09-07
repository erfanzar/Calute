---
name: authoring-tool-plugins
description: Build native Xerxes tool plugins with Bun tests and concrete usage examples in Plugin Creator mode.
---

# Native tool plugin authoring

1. Understand the requested tool and inspect the destination. Preserve existing files.
2. Create `.xerxes/plugins/<name>/index.ts`, `index.test.ts` and `README.md`.
3. Export `register(registry)` using the structural API below. Tool names use at most 57 letters, digits, underscores or hyphens. Register all tools under one plugin name.
4. Validate every argument. Callback arguments are positional `unknown` values; the model supplies `{ "args": [...] }`. Return serializable results. Keep registration free of filesystem, network and process side effects.
5. Test normal input, empty input and invalid input with `bun test <path>/index.test.ts`. For external operations use injected ports and cover errors and cancellation. Do not require live credentials for tests.
6. Document the install command, model request, exact tool arguments, expected result, disable/enable commands and prerequisites. Only claim verification actually performed.

```ts
interface Registry {
  registerTool(name: string, callback: (...args: unknown[]) => unknown,
    meta: { name: string; description: string; version: string }): void
}
export function register(registry: Registry): void {
  registry.registerTool('greet', (...args: unknown[]) => {
    if (args.length !== 1 || typeof args[0] !== 'string') {
      throw new Error('Expected one string argument')
    }
    return { greeting: `Hello, ${args[0]}!` }
  }, { name: 'greetings', description: 'Create a greeting', version: '1.0.0' })
}
```

After reviewing and testing, the user can run:

```text
/plugins install "/absolute/path/to/.xerxes/plugins/greetings/index.ts"
/plugins inspect greetings
```

Ask the model: “Call plugin_greet with args [\"Ada\"].” Expected result: `{"greeting":"Hello, Ada!"}`.

`/plugins disable greetings` disables new calls; `/plugins enable greetings` re-enables them. Source modules stay in place. After editing an installed module, restart Xerxes to reload its code. Installation executes the module; do not install as a side effect of authoring. Hooks, providers and channels require an embedding host and are not supported by managed installation. Never imply that a TypeScript plugin has a sandbox or cancellation signal the host does not provide.
