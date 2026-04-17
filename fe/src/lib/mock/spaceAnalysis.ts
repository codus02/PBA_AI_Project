import type { SpaceAnalysis } from '@/lib/types';

const MOCK_ANALYSES: SpaceAnalysis[] = [
  {
    style: '모던 라운지',
    mood: '세련되고 고급스러운',
    colors: ['딥 네이비', '골드', '크림화이트'],
    atmosphere: '조용하고 분위기 있는 저녁 모임에 어울리는 공간입니다.',
  },
  {
    style: '인더스트리얼 바',
    mood: '다이나믹하고 활기찬',
    colors: ['브릭 레드', '블랙', '로즈골드'],
    atmosphere: '활기차고 에너지 넘치는 파티에 최적화된 공간입니다.',
  },
  {
    style: '보태니컬 카페',
    mood: '편안하고 자연친화적인',
    colors: ['포레스트 그린', '아이보리', '테라코타'],
    atmosphere: '자연스럽고 따뜻한 느낌의 캐주얼 모임 공간입니다.',
  },
  {
    style: '루프탑 테라스',
    mood: '개방적이고 해방적인',
    colors: ['스카이블루', '화이트', '선셋 오렌지'],
    atmosphere: '탁 트인 뷰와 함께하는 특별한 파티를 위한 공간입니다.',
  },
  {
    style: '빈티지 재즈 클럽',
    mood: '낭만적이고 복고적인',
    colors: ['버건디', '앤틱 골드', '딥 브라운'],
    atmosphere: '재즈 선율이 흐르는 감성적인 소규모 파티 공간입니다.',
  },
];

export async function analyzeSpaceImage(_imageData: string): Promise<SpaceAnalysis> {
  // 실제 분석 대신 random mock 결과 반환
  await new Promise((resolve) => setTimeout(resolve, 1500));
  return MOCK_ANALYSES[Math.floor(Math.random() * MOCK_ANALYSES.length)];
}
