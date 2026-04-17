'use client';

import { useState } from 'react';
import type { DevicePayload } from '@/lib/types';
import Card, { CardHeader, CardTitle, CardBody } from '@/components/ui/Card';
import Button from '@/components/ui/Button';

interface DevicePreviewProps {
  payload: DevicePayload;
}

export default function DevicePreview({ payload }: DevicePreviewProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const jsonStr = JSON.stringify(payload, null, 2);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return null;
}
