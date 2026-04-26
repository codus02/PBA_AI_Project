'use client';

import { useState, useCallback, useRef } from 'react';
import type { SpaceAnalysis } from '@/lib/types';
import { uploadImg2Tag } from '@/lib/api';
import Button from '@/components/ui/Button';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';

interface SpaceUploadProps {
  onComplete: (analysis: SpaceAnalysis, imageUrl: string, file: File) => Promise<void> | void;
  onSkip: () => void;
}

export default function SpaceUpload({ onComplete, onSkip }: SpaceUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<SpaceAnalysis | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setSelectedFile(file);
    setAnalysis(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);

    setLoading(true);
    try {
      const data = await uploadImg2Tag(file);
      const tags = Array.isArray(data.mood_tag) ? data.mood_tag.filter(Boolean).slice(0, 3) : [];
      const [emotionTag = '', visualTag = '', spaceTag = ''] = tags;
      setAnalysis({
        emotionTag,
        visualTag,
        spaceTag,
        tags,
      });
    } catch (err) {
      setAnalysis(null);
      setError(err instanceof Error ? err.message : '이미지 분석에 실패했어요.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleApply = useCallback(async () => {
    if (!analysis || !preview || !selectedFile) return;

    setSubmitting(true);
    setError(null);
    try {
      await onComplete(analysis, preview, selectedFile);
    } catch (err) {
      setError(err instanceof Error ? err.message : '공간 이미지를 적용하지 못했어요.');
    } finally {
      setSubmitting(false);
    }
  }, [analysis, onComplete, preview, selectedFile]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>파티 공간 이미지 업로드</CardTitle>
          <p className="text-sm text-zinc-400 mt-1">
            파티 분위기에 맞는 칵테일을 추천해드려요
          </p>
        </CardHeader>
        <CardBody>
          {!preview ? (
            <div
              className={`flex flex-col items-center justify-center h-40 rounded-xl border-2 border-dashed transition-all ${
                dragOver
                  ? 'border-amber-400 bg-amber-400/10'
                  : 'border-zinc-700 bg-zinc-900/40'
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <span className="text-4xl mb-2">📷</span>
              <span className="text-sm text-zinc-300">사진을 촬영하거나 갤러리에서 선택하세요</span>
              <span className="text-xs text-zinc-500 mt-1">모바일에서는 카메라와 사진 보관함을 바로 열 수 있어요</span>
              <div className="flex gap-3 mt-4">
                <Button type="button" onClick={() => cameraInputRef.current?.click()}>
                  사진 촬영
                </Button>
                <Button variant="secondary" type="button" onClick={() => galleryInputRef.current?.click()}>
                  갤러리에서 선택
                </Button>
              </div>
              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                capture="environment"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                  e.currentTarget.value = '';
                }}
              />
              <input
                ref={galleryInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                  e.currentTarget.value = '';
                }}
              />
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="relative rounded-xl overflow-hidden h-40">
                <img
                  src={preview}
                  alt="공간 이미지"
                  className="w-full h-full object-cover"
                />
                {loading && (
                  <div className="absolute inset-0 bg-zinc-950/70 flex flex-col items-center justify-center gap-2">
                    <div className="w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm text-amber-400">공간 분석 중...</span>
                  </div>
                )}
              </div>

              {analysis && (
                <div className="flex flex-col gap-3">
                  <span className="text-green-400 text-sm">✓ 분석 완료</span>
                  <div className="flex flex-wrap gap-2">
                    {analysis.tags.map((tag) => (
                      <Badge key={tag} variant="amber">{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {error && (
                <p className="text-sm text-red-400">{error}</p>
              )}
            </div>
          )}
        </CardBody>
      </Card>

      <div className="flex gap-3">
        <Button variant="secondary" className="flex-1" onClick={onSkip} disabled={loading || submitting}>
          취소
        </Button>
        {analysis && preview && (
          <Button className="flex-1" onClick={handleApply} loading={submitting} disabled={loading || submitting}>
            이미지 적용하고 계속
          </Button>
        )}
      </div>
    </div>
  );
}
