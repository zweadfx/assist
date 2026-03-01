"use client";

import { useState } from "react";
import { type WeeklyRoutineResponse, type WeeklyDrill } from "@/lib/api";
import { type Language, t } from "@/lib/translations";

interface WeeklyRoutineCardProps {
  routine: WeeklyRoutineResponse;
  language: Language;
}

function PhaseIcon({ phase }: { phase: string }) {
  switch (phase) {
    case "warmup":
      return <span className="text-orange-500">&#9728;</span>;
    case "main":
      return <span className="text-red-500">&#9733;</span>;
    case "cooldown":
      return <span className="text-blue-500">&#10052;</span>;
    default:
      return null;
  }
}

function DrillCard({
  drill,
  language,
}: {
  drill: WeeklyDrill;
  language: Language;
}) {
  const [checked, setChecked] = useState(false);

  const phaseKey =
    drill.phase === "warmup"
      ? "phaseWarmup"
      : drill.phase === "main"
        ? "phaseMain"
        : "phaseCooldown";

  return (
    <div
      className={`rounded-lg border p-4 transition-colors ${
        checked ? "bg-green-50 border-green-300" : "bg-white border-gray-200"
      }`}
    >
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={checked}
          onChange={() => setChecked(!checked)}
          aria-label={drill.name}
          className="mt-1 h-4 w-4 rounded border-gray-300"
        />
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <PhaseIcon phase={drill.phase} />
            <span className="text-xs font-medium uppercase text-gray-500">
              {t(language, phaseKey)}
            </span>
            <span className="text-xs text-gray-400">
              {drill.duration_min} {t(language, "minutes")}
            </span>
            {drill.is_custom && (
              <span className="rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700">
                {t(language, "aiGenerated")}
              </span>
            )}
          </div>
          <h4
            className={`font-medium ${checked ? "line-through text-gray-400" : "text-gray-900"}`}
          >
            {drill.name}
          </h4>
          <p className="mt-1 text-sm text-gray-600">{drill.description}</p>
          <div className="mt-2 rounded-md bg-blue-50 p-2">
            <p className="text-xs text-blue-700">
              <span className="font-medium">
                {t(language, "coachingTip")}:
              </span>{" "}
              {drill.coaching_tip}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function WeeklyRoutineCard({
  routine,
  language,
}: WeeklyRoutineCardProps) {
  const [activeDay, setActiveDay] = useState(0);

  return (
    <div className="space-y-6">
      {/* Weekly Title & Overview */}
      <div className="rounded-lg bg-gradient-to-r from-blue-600 to-blue-800 p-6 text-white">
        <h2 className="text-2xl font-bold">{routine.weekly_title}</h2>
        <p className="mt-2 text-blue-100">{routine.coach_overview}</p>
      </div>

      {/* Day Tabs */}
      <div className="flex gap-1 overflow-x-auto rounded-lg bg-gray-100 p-1">
        {routine.days.map((day, index) => (
          <button
            key={day.day_number}
            onClick={() => setActiveDay(index)}
            className={`flex-1 whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              activeDay === index
                ? "bg-white text-blue-600 shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {t(language, "day")} {day.day_number}
          </button>
        ))}
      </div>

      {/* Active Day Content */}
      {routine.days[activeDay] && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">
              {routine.days[activeDay].day_label}
            </h3>
            <span className="text-sm text-gray-500">
              {t(language, "totalDuration")}:{" "}
              {routine.days[activeDay].total_duration_min}{" "}
              {t(language, "minutes")}
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {routine.days[activeDay].focus_areas.map((area) => (
              <span
                key={area}
                className="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700"
              >
                {area}
              </span>
            ))}
          </div>

          <div className="space-y-3">
            {routine.days[activeDay].drills.map((drill) => (
              <DrillCard
                key={drill.drill_id}
                drill={drill}
                language={language}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
