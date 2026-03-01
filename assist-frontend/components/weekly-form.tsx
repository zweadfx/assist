"use client";

import { useState } from "react";
import { type WeeklyRoutineRequest } from "@/lib/api";
import { type Language, t } from "@/lib/translations";

const FOCUS_OPTIONS = ["shooting", "dribble", "defense", "conditioning"] as const;
const EQUIPMENT_OPTIONS = [
  { value: "ball", labelKey: "ball" as const },
  { value: "hoop", labelKey: "hoop" as const },
  { value: "cones", labelKey: "cones" as const },
  { value: "jump-rope", labelKey: "jumpRope" as const },
  { value: "agility-ladder", labelKey: "agilityLadder" as const },
  { value: "resistance-band", labelKey: "resistanceBand" as const },
  { value: "plyo-box", labelKey: "plyoBox" as const },
];

interface WeeklyFormProps {
  onSubmit: (request: WeeklyRoutineRequest) => void;
  isLoading: boolean;
  language: Language;
  onLanguageChange: (lang: Language) => void;
}

export default function WeeklyForm({
  onSubmit,
  isLoading,
  language,
  onLanguageChange,
}: WeeklyFormProps) {
  const [skillLevel, setSkillLevel] = useState<
    "beginner" | "intermediate" | "advanced"
  >("intermediate");
  const [trainingDays, setTrainingDays] = useState(3);
  const [focusAreas, setFocusAreas] = useState<string[]>(["shooting"]);
  const [availableTime, setAvailableTime] = useState(60);
  const [equipment, setEquipment] = useState<string[]>(["ball"]);
  const [freeText, setFreeText] = useState("");

  const toggleFocusArea = (area: string) => {
    setFocusAreas((prev) =>
      prev.includes(area)
        ? prev.length > 1
          ? prev.filter((a) => a !== area)
          : prev
        : [...prev, area]
    );
  };

  const toggleEquipment = (item: string) => {
    setEquipment((prev) =>
      prev.includes(item)
        ? prev.filter((e) => e !== item)
        : [...prev, item]
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const guardedTime = Number.isFinite(availableTime) && availableTime > 0
      ? availableTime
      : 60;
    onSubmit({
      skill_level: skillLevel,
      training_days: trainingDays,
      focus_areas: focusAreas as WeeklyRoutineRequest["focus_areas"],
      available_time_per_day_min: guardedTime,
      equipment,
      language,
      free_text: freeText || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Language Toggle */}
      <div className="flex justify-end">
        <select
          value={language}
          onChange={(e) => onLanguageChange(e.target.value as Language)}
          className="rounded-md border border-gray-300 px-3 py-1 text-sm"
        >
          <option value="en">English</option>
          <option value="ko">한국어</option>
        </select>
      </div>

      {/* Skill Level */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "skillLevel")}
        </label>
        <div className="flex gap-2">
          {(["beginner", "intermediate", "advanced"] as const).map((level) => (
            <button
              key={level}
              type="button"
              onClick={() => setSkillLevel(level)}
              className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                skillLevel === level
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {t(language, level)}
            </button>
          ))}
        </div>
      </div>

      {/* Training Days */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "trainingDays")}: {trainingDays}
        </label>
        <input
          type="range"
          min={1}
          max={7}
          value={trainingDays}
          onChange={(e) => setTrainingDays(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500">
          {[1, 2, 3, 4, 5, 6, 7].map((d) => (
            <span key={d}>{d}</span>
          ))}
        </div>
      </div>

      {/* Focus Areas (multi-select) */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "focusAreas")}
        </label>
        <div className="flex flex-wrap gap-2">
          {FOCUS_OPTIONS.map((area) => (
            <button
              key={area}
              type="button"
              onClick={() => toggleFocusArea(area)}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                focusAreas.includes(area)
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {t(language, area)}
            </button>
          ))}
        </div>
      </div>

      {/* Available Time Per Day */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "availableTimePerDay")}
        </label>
        <input
          type="number"
          min={15}
          max={180}
          value={availableTime}
          onChange={(e) => setAvailableTime(Number(e.target.value))}
          className="w-full rounded-md border border-gray-300 px-3 py-2"
        />
      </div>

      {/* Equipment */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "equipment")}
        </label>
        <div className="flex flex-wrap gap-2">
          {EQUIPMENT_OPTIONS.map(({ value, labelKey }) => (
            <button
              key={value}
              type="button"
              onClick={() => toggleEquipment(value)}
              className={`rounded-full px-3 py-1.5 text-sm transition-colors ${
                equipment.includes(value)
                  ? "bg-green-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {t(language, labelKey)}
            </button>
          ))}
        </div>
      </div>

      {/* Free Text */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {t(language, "freeText")}
        </label>
        <textarea
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          maxLength={500}
          rows={3}
          placeholder={t(language, "freeTextPlaceholder")}
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
        />
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading}
        className="w-full rounded-md bg-blue-600 px-4 py-3 text-white font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? t(language, "loading") : t(language, "generateWeekly")}
      </button>
    </form>
  );
}
