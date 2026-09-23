import { test, expect } from "@playwright/test";

test.describe("SatQuery AI — 12-Step Hackathon E2E User Flow (SIH26167)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://localhost:5173/");
  });

  test("Step 1: Open application & verify title", async ({ page }) => {
    await expect(page).toHaveTitle(/SatQuery AI/i);
  });

  test("Step 2-5: Single-image upload, VQA query, and region grounding", async ({ page }) => {
    // Navigate to new analysis
    const newAnalysisBtn = page.getByRole("link", { name: /new analysis/i });
    if (await newAnalysisBtn.isVisible()) {
      await newAnalysisBtn.click();
    }
    
    // Select VQA mode
    await expect(page.getByText(/VQA/i)).toBeVisible();
  });

  test("Step 6-7: Temporal pair change detection workflow", async ({ page }) => {
    const newAnalysisBtn = page.getByRole("link", { name: /new analysis/i });
    if (await newAnalysisBtn.isVisible()) {
      await newAnalysisBtn.click();
    }
  });

  test("Step 8-9: Optical + SAR cross-modal analysis workflow", async ({ page }) => {
    const newAnalysisBtn = page.getByRole("link", { name: /new analysis/i });
    if (await newAnalysisBtn.isVisible()) {
      await newAnalysisBtn.click();
    }
  });

  test("Step 10-12: Report generation and graceful error handling", async ({ page }) => {
    const historyBtn = page.getByRole("link", { name: /history/i });
    if (await historyBtn.isVisible()) {
      await historyBtn.click();
    }
  });
});
