'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Party,
  Guest,
  GuestStep,
  GuestPreferences,
  FollowUpAnswer,
  CocktailRecommendation,
  FeedbackAnalysis,
  BrewState,
  BrewStatus,
  SpaceAnalysis,
  GuestLog,
  ConversationEntry,
  RecommendationEntry,
  FeedbackEntry,
} from '@/lib/types';
import { generateId, generatePartyCode, now } from '@/lib/utils';

// ─────────────────────────────────────────
// Store Interface
// ─────────────────────────────────────────
interface PartyStore {
  parties: Record<string, Party>;

  // Party actions
  createParty: (name: string) => Party;
  getParty: (id: string) => Party | undefined;
  setSpaceAnalysis: (partyId: string, analysis: SpaceAnalysis) => void;
  setSpaceImage: (partyId: string, imageUrl: string) => void;

  // Guest actions
  addGuest: (partyId: string, name: string) => Guest;
  getGuest: (partyId: string, guestId: string) => Guest | undefined;
  updateGuestStep: (partyId: string, guestId: string, step: GuestStep) => void;
  setPreferences: (partyId: string, guestId: string, prefs: GuestPreferences) => void;
  setFollowUpAnswers: (partyId: string, guestId: string, answers: FollowUpAnswer[]) => void;
  setTastingRecommendation: (partyId: string, guestId: string, rec: CocktailRecommendation) => void;
  setFeedback: (partyId: string, guestId: string, feedback: string) => void;
  setFeedbackAnalysis: (partyId: string, guestId: string, analysis: FeedbackAnalysis) => void;
  setFinalRecommendation: (partyId: string, guestId: string, rec: CocktailRecommendation) => void;
  updateBrewState: (partyId: string, guestId: string, brewState: Partial<BrewState>) => void;
  setSatisfaction: (partyId: string, guestId: string, score: number) => void;

  // DB ID actions
  setPartyDbId: (partyId: string, dbId: string) => void;
  setGuestDbId: (partyId: string, guestId: string, dbId: string) => void;
  setSampleRecommendationId: (partyId: string, guestId: string, id: string) => void;
  setFinalRecommendationId: (partyId: string, guestId: string, id: string) => void;

  // Log actions
  addConversationEntry: (partyId: string, guestId: string, entry: Omit<ConversationEntry, 'id'>) => void;
  addRecommendationEntry: (partyId: string, guestId: string, entry: Omit<RecommendationEntry, 'id'>) => void;
  addFeedbackEntry: (partyId: string, guestId: string, entry: Omit<FeedbackEntry, 'id'>) => void;
}

// ─────────────────────────────────────────
// Helper
// ─────────────────────────────────────────
function makeEmptyLog(): GuestLog {
  return {
    conversationLog: [],
    recommendationLog: [],
    feedbackLog: [],
  };
}

function updateGuest(
  parties: Record<string, Party>,
  partyId: string,
  guestId: string,
  updater: (g: Guest) => Guest
): Record<string, Party> {
  const party = parties[partyId];
  if (!party) return parties;
  return {
    ...parties,
    [partyId]: {
      ...party,
      guests: party.guests.map((g) => (g.id === guestId ? updater(g) : g)),
    },
  };
}

// ─────────────────────────────────────────
// Store
// ─────────────────────────────────────────
export const usePartyStore = create<PartyStore>()(
  persist(
    (set, get) => ({
      parties: {},

      createParty: (name) => {
        const party: Party = {
          id: generateId(),
          code: generatePartyCode(),
          name,
          guests: [],
          createdAt: now(),
        };
        set((s) => ({ parties: { ...s.parties, [party.id]: party } }));
        return party;
      },

      getParty: (id) => get().parties[id],

      setSpaceAnalysis: (partyId, analysis) =>
        set((s) => ({
          parties: {
            ...s.parties,
            [partyId]: { ...s.parties[partyId], spaceAnalysis: analysis },
          },
        })),

      setSpaceImage: (partyId, imageUrl) =>
        set((s) => ({
          parties: {
            ...s.parties,
            [partyId]: { ...s.parties[partyId], spaceImage: imageUrl },
          },
        })),

      addGuest: (partyId, name) => {
        const guest: Guest = {
          id: generateId(),
          partyId,
          name,
          step: 'onboarding',
          logs: makeEmptyLog(),
        };
        set((s) => ({
          parties: {
            ...s.parties,
            [partyId]: {
              ...s.parties[partyId],
              guests: [...s.parties[partyId].guests, guest],
            },
          },
        }));
        return guest;
      },

      getGuest: (partyId, guestId) =>
        get().parties[partyId]?.guests.find((g) => g.id === guestId),

      updateGuestStep: (partyId, guestId, step) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({ ...g, step })),
        })),

      setPreferences: (partyId, guestId, preferences) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            preferences,
          })),
        })),

      setFollowUpAnswers: (partyId, guestId, followUpAnswers) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            followUpAnswers,
          })),
        })),

      setTastingRecommendation: (partyId, guestId, tastingRecommendation) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            tastingRecommendation,
            step: 'tasting',
          })),
        })),

      setFeedback: (partyId, guestId, feedback) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            feedback,
            step: 'feedback',
          })),
        })),

      setFeedbackAnalysis: (partyId, guestId, feedbackAnalysis) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            feedbackAnalysis,
          })),
        })),

      setFinalRecommendation: (partyId, guestId, finalRecommendation) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            finalRecommendation,
            step: 'final',
          })),
        })),

      updateBrewState: (partyId, guestId, brewState) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            brewState: { ...g.brewState, ...brewState } as BrewState,
            step: brewState.status === 'complete' ? 'complete' : g.step,
          })),
        })),

      setSatisfaction: (partyId, guestId, score) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            satisfaction: score,
            logs: { ...g.logs, satisfaction: score },
          })),
        })),

      setPartyDbId: (partyId, dbId) =>
        set((s) => ({
          parties: {
            ...s.parties,
            [partyId]: { ...s.parties[partyId], dbId },
          },
        })),

      setGuestDbId: (partyId, guestId, dbId) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({ ...g, dbId })),
        })),

      setSampleRecommendationId: (partyId, guestId, id) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({ ...g, sampleRecommendationId: id })),
        })),

      setFinalRecommendationId: (partyId, guestId, id) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({ ...g, finalRecommendationId: id })),
        })),

      addConversationEntry: (partyId, guestId, entry) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            logs: {
              ...g.logs,
              conversationLog: [
                ...g.logs.conversationLog,
                { ...entry, id: generateId() },
              ],
            },
          })),
        })),

      addRecommendationEntry: (partyId, guestId, entry) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            logs: {
              ...g.logs,
              recommendationLog: [
                ...g.logs.recommendationLog,
                { ...entry, id: generateId() },
              ],
            },
          })),
        })),

      addFeedbackEntry: (partyId, guestId, entry) =>
        set((s) => ({
          parties: updateGuest(s.parties, partyId, guestId, (g) => ({
            ...g,
            logs: {
              ...g.logs,
              feedbackLog: [
                ...g.logs.feedbackLog,
                { ...entry, id: generateId() },
              ],
            },
          })),
        })),
    }),
    {
      name: 'party-store',
      partialize: (state) => ({
        ...state,
        parties: Object.fromEntries(
          Object.entries(state.parties).map(([id, party]) => [
            id,
            { ...party, spaceImage: undefined },
          ])
        ),
      }),
    }
  )
);

// ─────────────────────────────────────────
// Selector hooks
// ─────────────────────────────────────────
export const useParty = (partyId: string) =>
  usePartyStore((s) => s.parties[partyId]);

export const useGuest = (partyId: string, guestId: string) =>
  usePartyStore((s) => s.parties[partyId]?.guests.find((g) => g.id === guestId));
