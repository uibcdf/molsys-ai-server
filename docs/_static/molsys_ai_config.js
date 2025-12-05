// Default configuration for the MolSys-AI docs chat widget (pilot).
// You can override these values in the HTML pages (or another script) by
// assigning to `window.molsysAiChatConfig` before the widget script runs.

window.molsysAiChatConfig = window.molsysAiChatConfig || {
  // "placeholder": always show the friendly placeholder reply.
  // "backend": call the docs-chat backend and show its responses.
  mode: "placeholder",
  // By default, talk to a docs-chat backend on the same origin.
  backendUrl: window.location.origin.replace(/\/+$/, "") + "/v1/docs-chat",
};

