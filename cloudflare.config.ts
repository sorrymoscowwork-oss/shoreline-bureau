import { defineConfig } from "cf/config";

export default defineConfig({
  accountId: "80ac49029f001957ff20b3f9428be161",
  worker: {
    name: "shoreline-bureau",
    compatibilityDate: "2026-11-16",
    assets: {
      directory: "./public",
      notFoundHandling: "404-page",
    },
  },
});
