
// Minimal MolSys-AI documentation chat widget (pilot).
//
// For the first pilot, the widget:
// - renders a small chat box on pages that include it,
// - captures user messages,
// - always responds with a friendly placeholder message.
//
// No network requests are made yet; integration with the real docs_chat
// backend will be added in later iterations.

(function () {
  function ensureString(value) {
    return typeof value === "string" ? value : "";
  }

  function createChatContainer(root) {
    const globalConfig = window.molsysAiChatConfig || {};
    const mode = globalConfig.mode || "placeholder"; // "placeholder" | "backend"
    const backendUrl =
      globalConfig.backendUrl ||
      (window.location.origin.replace(/\/+$/, "") + "/v1/docs-chat");
    const container = document.createElement("div");
    container.style.border = "1px solid #ccc";
    container.style.borderRadius = "6px";
    container.style.maxWidth = "400px";
    container.style.height = "320px";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    container.style.fontSize = "14px";
    container.style.backgroundColor = "#fafafa";

    const header = document.createElement("div");
    header.textContent = "MolSys-AI Docs Helper (pilot)";
    header.style.padding = "8px 10px";
    header.style.borderBottom = "1px solid #ddd";
    header.style.fontWeight = "bold";
    header.style.backgroundColor = "#f0f0f0";

    const messages = document.createElement("div");
    messages.style.flex = "1";
    messages.style.padding = "8px";
    messages.style.overflowY = "auto";

    const inputRow = document.createElement("div");
    inputRow.style.display = "flex";
    inputRow.style.borderTop = "1px solid #ddd";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Ask something about the docs...";
    input.style.flex = "1";
    input.style.border = "none";
    input.style.padding = "8px";
    input.style.outline = "none";
    input.setAttribute("aria-label", "MolSys-AI docs chatbot input");

    const button = document.createElement("button");
    button.textContent = "Send";
    button.style.border = "none";
    button.style.padding = "0 12px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#4a8af4";
    button.style.color = "#fff";
    button.style.fontWeight = "bold";

    inputRow.appendChild(input);
    inputRow.appendChild(button);

    container.appendChild(header);
    container.appendChild(messages);
    container.appendChild(inputRow);

    root.appendChild(container);

    function appendMessage(text, role) {
      const bubble = document.createElement("div");
      bubble.textContent = text;
      bubble.style.margin = "4px 0";
      bubble.style.padding = "6px 8px";
      bubble.style.borderRadius = "4px";

      if (role === "user") {
        bubble.style.alignSelf = "flex-end";
        bubble.style.backgroundColor = "#dbeafe";
      } else {
        bubble.style.alignSelf = "flex-start";
        bubble.style.backgroundColor = "#e5e7eb";
      }

      messages.appendChild(bubble);
      messages.scrollTop = messages.scrollHeight;
    }

    const conversation = [];
    const greeting =
      mode === "backend"
        ? "Hi! Ask me about the MolSys* documentation."
        : "Hi! This MolSys-AI docs bot is not ready yet. Stay tuned for more features soon...";
    appendMessage(greeting, "assistant");
    conversation.push({ role: "assistant", content: greeting });

    function handleSend() {
      const text = input.value.trim();
      if (!text) return;
      appendMessage(text, "user");
      input.value = "";
      conversation.push({ role: "user", content: text });

      if (mode === "backend") {
        // Call the docs-chat backend. Even in backend mode this stays simple:
        // no streaming, just a single request/response.
        try {
          fetch(backendUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: conversation, k: 5 }),
          })
            .then(function (resp) {
              if (!resp.ok) {
                throw new Error("HTTP " + resp.status);
              }
              return resp.json();
            })
            .then(function (data) {
              var answer = ensureString(data && (data.answer || data.content));
              if (!answer) {
                answer = "The docs backend did not return a valid answer.";
              }
              appendMessage(answer, "assistant");
              conversation.push({ role: "assistant", content: answer });
            })
            .catch(function (err) {
              console.error("MolSys-AI docs widget error:", err);
              var fallback =
                "The docs backend is not available right now. "
                  + "Please try again later.";
              appendMessage(fallback, "assistant");
              conversation.push({ role: "assistant", content: fallback });
            });
        } catch (e) {
          console.error("MolSys-AI docs widget error:", e);
          var fallback =
            "The docs backend is not available right now. "
              + "Please try again later.";
          appendMessage(fallback, "assistant");
          conversation.push({ role: "assistant", content: fallback });
        }
      } else {
        // Placeholder mode: always respond with the same friendly message.
        var placeholder =
          "This bot is still under development. "
            + "In future versions it will answer questions about the MolSys* documentation.";
        appendMessage(placeholder, "assistant");
        conversation.push({ role: "assistant", content: placeholder });
      }
    }

    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        handleSend();
      }
    });
    button.addEventListener("click", handleSend);
  }

  function bootstrap() {
    var explicitRoot = document.getElementById("molsys-ai-chat");
    if (explicitRoot) {
      // Inline mode: render the chat box directly inside the given container.
      createChatContainer(explicitRoot);
    } else {
      // Fallback mode: create a floating launcher + panel in the bottom-right
      // corner of the page.
      var wrapper = document.createElement("div");
      wrapper.style.position = "fixed";
      wrapper.style.bottom = "16px";
      wrapper.style.right = "16px";
      wrapper.style.zIndex = "9999";
      wrapper.style.display = "flex";
      wrapper.style.flexDirection = "column";
      wrapper.style.alignItems = "flex-end";

      var panelHost = document.createElement("div");
      panelHost.style.marginBottom = "8px";
      panelHost.style.display = "none";

      createChatContainer(panelHost);

      var button = document.createElement("button");
      button.setAttribute("type", "button");
      button.setAttribute("aria-label", "Open MolSys-AI docs helper");
      button.textContent = "AI";
      button.style.width = "40px";
      button.style.height = "40px";
      button.style.borderRadius = "999px";
      button.style.border = "none";
      button.style.backgroundColor = "#4a8af4";
      button.style.color = "#fff";
      button.style.fontWeight = "bold";
      button.style.cursor = "pointer";
      button.style.boxShadow = "0 2px 6px rgba(0, 0, 0, 0.2)";

      var isOpen = false;
      function togglePanel() {
        isOpen = !isOpen;
        panelHost.style.display = isOpen ? "block" : "none";
        button.setAttribute(
          "aria-label",
          isOpen ? "Close MolSys-AI docs helper" : "Open MolSys-AI docs helper"
        );
      }

      button.addEventListener("click", togglePanel);

      wrapper.appendChild(panelHost);
      wrapper.appendChild(button);
      document.body.appendChild(wrapper);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }

  console.log("MolSys-AI docs widget loaded (pilot).");
})();
