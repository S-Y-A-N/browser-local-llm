# LLM WebGPU Chat Interface

## Overview

This repository provides a lightweight, client-side web application for running Qwen3 large language models entirely within the browser. Built on the Wllama engine, the application utilizes WebAssembly and WebGPU to execute local inference of quantized GGUF models without relying on server-side processing or external API dependencies. 

A live demonstration of this project is available at: [asubah.github.io/browserllm](https://asubah.github.io/browserllm).

## Features

* **Client-Side Inference:** Executes models directly in the user's browser, ensuring data privacy and zero server compute costs.
* **Hardware Acceleration:** Automatically detects and utilizes WebGPU for accelerated processing, with a seamless fallback to CPU execution if unavailable.
* **Persistent Model Caching:** Leverages the Origin Private File System (OPFS) to cache downloaded model shards, significantly reducing load times for subsequent sessions.
* **Streaming Generation:** Implements real-time token streaming and dynamic rendering of inference output.
* **Model Flexibility:** Supports seamless switching between multiple [Qwen3](https://huggingface.co/asubah/Qwen3-GGUF) parameter sizes (0.6B, 1.7B, and 4B).

## Dependencies

* [@wllama/wllama](https://github.com/ngxson/wllama): WebAssembly framework for LLaMA inference.
* [Tailwind CSS](https://tailwindcss.com/): CSS framework.
