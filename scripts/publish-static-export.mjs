import { cp, mkdir } from "node:fs/promises";

await mkdir("public", { recursive: true });
await cp("out", "public", { recursive: true, force: true });

