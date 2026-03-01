"use client";

import { useState } from "react";
import WeeklyForm from "@/components/weekly-form";
import WeeklyRoutineCard from "@/components/weekly-routine-card";
import {
  postWeeklyRoutine,
  type WeeklyRoutineRequest,
  type WeeklyRoutineResponse,
} from "@/lib/api";
import { type Language, t } from "@/lib/translations";

export default function WeeklyPage() {
  const [language, setLanguage] = useState<Language>("en");
  const [isLoading, setIsLoading] = useState(false);
  const [routine, setRoutine] = useState<WeeklyRoutineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (request: WeeklyRoutineRequest) => {
    setIsLoading(true);
    setError(null);
    setRoutine(null);

    try {
      const result = await postWeeklyRoutine(request);
      setRoutine(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : t(language, "error"));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-900">
          {t(language, "weeklyTitle")}
        </h1>
        <p className="mt-2 text-gray-600">
          {t(language, "weeklySubtitle")}
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[400px_1fr]">
        {/* Form Section */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
          <WeeklyForm
            onSubmit={handleSubmit}
            isLoading={isLoading}
            language={language}
            onLanguageChange={setLanguage}
          />
        </div>

        {/* Results Section */}
        <div className="min-w-0">
          {isLoading && (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />
                <p className="mt-4 text-gray-600">{t(language, "loading")}</p>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {routine && !isLoading && (
            <WeeklyRoutineCard routine={routine} language={language} />
          )}
        </div>
      </div>
    </div>
  );
}
