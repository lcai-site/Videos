
import React, { useState, useMemo, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality, Type } from "@google/genai";

// --- CONSTANTS ---

const FONT_FAMILIES = [
    { name: 'Anton', value: "'Anton', sans-serif" },
    { name: 'Oswald', value: "'Oswald', sans-serif" },
    { name: 'Bebas Neue', value: "'Bebas Neue', cursive" },
    { name: 'Roboto Condensed', value: "'Roboto Condensed', sans-serif" },
    { name: 'Montserrat', value: "'Montserrat', sans-serif" },
    { name: 'Poppins', value: "'Poppins', sans-serif" },
    { name: 'Lobster', value: "'Lobster', cursive" },
    { name: 'Pacifico', value: "'Pacifico', cursive" },
    { name: 'Archivo Black', value: "'Archivo Black', sans-serif" },
    { name: 'Passion One', value: "'Passion One', cursive" },
    { name: 'Alfa Slab One', value: "'Alfa Slab One', cursive" },
    { name: 'Black Ops One', value: "'Black Ops One', cursive" },
    { name: 'Bangers', value: "'Bangers', cursive" },
    { name: 'Permanent Marker', value: "'Permanent Marker', cursive" },
    { name: 'Ultra', value: "'Ultra', serif" },
    { name: 'Luckiest Guy', value: "'Luckiest Guy', cursive" },
    { name: 'Righteous', value: "'Righteous', cursive" },
    { name: 'Staatliches', value: "'Staatliches', cursive" },
    { name: 'Patua One', value: "'Patua One', cursive" },
    { name: 'Changa', value: "'Changa', sans-serif" },
    { name: 'Inter', value: 'Inter, sans-serif' },
    { name: 'Roboto', value: "'Roboto', sans-serif" },
    { name: 'Lato', value: "'Lato', sans-serif" },
    { name: 'Open Sans', value: "'Open Sans', sans-serif" },
    { name: 'Source Sans Pro', value: "'Source Sans Pro', sans-serif" },
    { name: 'Raleway', value: "'Raleway', sans-serif" },
];

const VOICES = [
    { id: 'Zephyr', name: 'Feminina 1' },
    { id: 'Charon', name: 'Feminina 2' },
    { id: 'Kore', name: 'Masculina 1' },
    { id: 'Puck', name: 'Masculina 2' },
    { id: 'Fenrir', name: 'Masculina 3' },
];

const LANGUAGES = [
    { code: 'pt', name: 'Português' },
    { code: 'en', name: 'Inglês' },
    { code: 'es', name: 'Espanhol' },
    { code: 'fr', name: 'Francês' },
    { code: 'de', name: 'Alemão' },
    { code: 'it', name: 'Italiano' },
];

// --- TYPE DEFINITIONS ---

// For Video Editor
interface VideoElement {
  id: string;
  type: 'text';
  content: string;
  x: number; // 0-100 percentage
  y: number; // 0-100 percentage
  size: number; // represents font size
  color: string;
  bgColor: string;
  hasBg: boolean;
  fontFamily: string;
  fontWeight: string;
  fontStyle: string;
  textTransform: string;
  strokeColor: string;
  strokeWidth: number;
  hasStroke: boolean;
  textAlign: 'center' | 'left' | 'right';
  aspectRatio: number;
}

// For Batch Video Editor
interface VideoSlot {
  id: number;
  file: File | null;
  url: string | null;
  thumbnail: string | null;
  lang: string;
}


// For Image Creator
interface AnalysisResult {
    language: string;
    translation: string;
    text_elements: {
        title: string;
        points: string[];
        secondary_text: string[];
        cta: string;
    };
    visual_description: string;
}

// For Image Creator
interface GeneratedImage {
    src: string;
    translation?: string;
}

// --- HELPER FUNCTIONS ---

const fileToGenerativePart = async (file: File) => {
    const base64EncodedDataPromise = new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(file);
    });
    return {
        inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
    };
};

function decodeBase64(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodePcmAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000,
  numChannels: number = 1,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

async function concatenateAudioBuffers(buffers: AudioBuffer[], context: AudioContext): Promise<AudioBuffer> {
    if (buffers.length === 0) {
        const emptyBuffer = context.createBuffer(1, 1, context.sampleRate);
        context.close();
        return emptyBuffer;
    }
    
    const numberOfChannels = Math.max(...buffers.map(b => b.numberOfChannels));
    const totalLength = buffers.reduce((sum, b) => sum + b.length, 0);
    
    const newBuffer = context.createBuffer(numberOfChannels, totalLength, context.sampleRate);
    
    let offset = 0;
    for (const buffer of buffers) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            // If source buffer has fewer channels, reuse its first channel
            const sourceChannel = Math.min(channel, buffer.numberOfChannels - 1);
            const channelData = buffer.getChannelData(sourceChannel);
            newBuffer.copyToChannel(channelData, channel, offset);
        }
        offset += buffer.length;
    }
    context.close();
    return newBuffer;
}


// --- SVG ICONS ---

const VideoIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="m22 8-6 4 6 4V8Z" /><rect width="14" height="12" x="2" y="6" rx="2" ry="2" /></svg>
);

const ImageIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><rect width="18" height="18" x="3" y="3" rx="2" ry="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" /></svg>
);

// FIX: Corrected the viewBox attribute which was malformed and causing parsing errors.
const UploadIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" /></svg>
);

const TrashIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" {...props}><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
);

const ExportIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M12 17v-10"/><path d="m15 10-3-3-3 3"/><path d="M19 21H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2z"/></svg>
);

const SpeakerOnIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>
);

const SpeakerOffIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="1" x2="1" y2="23"></line><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
);

const ChangeVideoIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M10.03 16.03 8 18l2.03-1.97"/><path d="M8 12.5v5.5"/><path d="m13.97 11.97 2.03-1.97L14 8"/><path d="M16 15.5v-5.5"/></svg>
);

const TextIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M17 6.1H7c-1.1 0-2 .9-2 2v7.8c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V8.1c0-1.1-.9-2-2-2z"></path><path d="M12 18v-5.5"></path><path d="M8 18v-5.5"></path><path d="M16 18v-5.5"></path><path d="M10 12.5h4"></path></svg>
);

const GalleryIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
);

const NarrationIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
);

const ClockIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
);

// --- FEATURE MODULES ---
// These are the main, high-level features of the application.

/**
 * The Image Creator module.
 * Analyzes a reference creative and generates AI-powered variations.
 */
const ImageCreator = () => {
    const [referenceFile, setReferenceFile] = useState<File | null>(null);
    const [referencePreview, setReferencePreview] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
    const ai = useMemo(() => new GoogleGenAI({ apiKey: process.env.API_KEY }), []);

    const resetState = () => {
        setReferenceFile(null);
        setReferencePreview(null);
        setIsAnalyzing(false);
        setAnalysis(null);
        setIsLoading(false);
        setError(null);
        setGeneratedImages([]);
    };

    const analyzeCreative = async (file: File) => {
        resetState();
        setReferenceFile(file);
        setReferencePreview(URL.createObjectURL(file));
        setIsAnalyzing(true);
        setError(null);

        try {
            const imagePart = await fileToGenerativePart(file);
            const analysisSchema = {
                type: Type.OBJECT,
                properties: {
                    language: { type: Type.STRING, description: "O código ISO 639-1 para o idioma detectado do texto na imagem (ex: 'en', 'pt', 'es')." },
                    translation: { type: Type.STRING, description: "Se o idioma não for Português ('pt'), forneça uma tradução completa e precisa de todo o texto para o Português do Brasil. Caso contrário, retorne uma string vazia." },
                    text_elements: {
                        type: Type.OBJECT,
                        description: "Um objeto contendo o texto extraído, categorizado por sua função no anúncio.",
                        properties: {
                            title: { type: Type.STRING, description: "O título ou manchete principal do anúncio." },
                            points: { type: Type.ARRAY, description: "Uma lista de pontos-chave ou características.", items: { type: Type.STRING } },
                            secondary_text: { type: Type.ARRAY, description: "Quaisquer linhas secundárias de texto ou subtítulos.", items: { type: Type.STRING } },
                            cta: { type: Type.STRING, description: "A chamada para ação (call to action) final." }
                        },
                        required: ['title', 'points', 'secondary_text', 'cta']
                    },
                    visual_description: { type: Type.STRING, description: "Uma descrição detalhada do estilo visual do anúncio, incluindo fundo, cores, fontes e atmosfera geral. Exemplo: 'Design minimalista com fundo de gradiente cinza claro e padrões de penas. O título usa uma fonte serifada vermelha e em negrito. Os pontos usam uma fonte sans-serif preta padrão. O texto secundário está em azul.'" }
                },
                required: ['language', 'translation', 'text_elements', 'visual_description']
            };

            const prompt = "Analise este criativo de anúncio. Extraia todos os elementos de texto e categorize-os. Descreva o estilo visual, cores e tema. Se o texto não estiver em Português do Brasil, traduza-o. Retorne o resultado como um objeto JSON.";

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: { parts: [imagePart, { text: prompt }] },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: analysisSchema
                }
            });

            const result = JSON.parse(response.text) as AnalysisResult;
            setAnalysis(result);

        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'Falha ao analisar o criativo.');
            setAnalysis(null);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            analyzeCreative(file);
        }
    };

    const generateVariations = async () => {
        if (!analysis) {
            setError('A análise do criativo deve ser concluída antes de gerar variações.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setGeneratedImages([]);

        try {
            const { visual_description, text_elements, language } = analysis;
            const originalLanguageName = new Intl.DisplayNames(['pt-BR'], { type: 'language' }).of(language) || language;
            
            const copyGenerationSchema = {
                type: Type.OBJECT,
                properties: {
                    variations: {
                        type: Type.ARRAY,
                        description: "Um array de 5 variações distintas de texto de anúncio.",
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                title: { type: Type.STRING, description: "O novo título cativante." },
                                points: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Uma lista de pontos-chave ou características." },
                                cta: { type: Type.STRING, description: "A chamada para ação final." }
                            },
                            required: ['title', 'points', 'cta']
                        }
                    }
                },
                required: ['variations']
            };

            const copyPrompt = `
                Analisando o seguinte texto de um anúncio, sua primeira tarefa é identificar e corrigir quaisquer erros de ortografia ou gramática.
                Depois, com base na mensagem corrigida, crie 5 novas variações de copy (texto publicitário).

                **Texto Original (para correção e inspiração):**
                - Idioma: ${originalLanguageName} (${language})
                - Título: "${text_elements.title}"
                - Pontos-chave: ${text_elements.points.map(p => `"${p}"`).join(', ')}
                - CTA: "${text_elements.cta}"

                **Requisitos para as 5 Variações:**
                1. **Idioma:** Todas as variações DEVEM estar em ${originalLanguageName}.
                2. **Qualidade:** O texto deve ser gramaticalmente perfeito, persuasivo e profissional.
                3. **Estrutura:** Mantenha uma estrutura semelhante (título, ${text_elements.points.length > 0 ? text_elements.points.length : 'alguns'} pontos, cta).
                4. **Originalidade:** As variações devem ser distintas umas das outras.

                Retorne um objeto JSON contendo as 5 variações.
            `;

            const textResponse = await ai.models.generateContent({
                model: 'gemini-2.5-pro',
                contents: { parts: [{ text: copyPrompt }] },
                config: {
                    responseMimeType: "application/json",
                    responseSchema: copyGenerationSchema,
                }
            });
            
            const result = JSON.parse(textResponse.text);
            const copyVariations = result.variations;

            if (!copyVariations || copyVariations.length === 0) {
                throw new Error("A IA não conseguiu gerar variações de texto.");
            }

            const imageGenerationPromises = copyVariations.slice(0, 5).map((copy: any) => {
                const imagePrompt = `
                    Crie um criativo de anúncio profissional e moderno com uma imagem de fundo relevante.
                    **Descrição Visual de Referência:** "${visual_description}". Crie um novo design inspirado nisso, mas não idêntico.
                    
                    **Incorpore o seguinte texto na imagem de forma clara, legível e com boa hierarquia visual:**
                    - **Título (destaque principal):** "${copy.title}"
                    - **Pontos-chave:** ${copy.points.map((p:string) => `\n  - ${p}`).join('')}
                    - **Call to Action (CTA):** "${copy.cta}"

                    **Requisitos CRÍTICOS:**
                    - A imagem DEVE ter uma proporção de **1:1 (quadrada)**.
                    - TODO o texto renderizado na imagem deve ser **exatamente** como fornecido, sem erros de ortografia.
                    - Organize os elementos de forma limpa e profissional.
                `;
                
                return ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: { parts: [{ text: imagePrompt }] },
                    config: {
                        responseModalities: [Modality.IMAGE],
                    },
                });
            });

            const imageResponses = await Promise.all(imageGenerationPromises);

            const images = imageResponses.flatMap(response =>
                (response.candidates?.[0]?.content?.parts ?? [])
                    .filter(part => part.inlineData)
                    .map(part => `data:${part.inlineData!.mimeType};base64,${part.inlineData!.data}`)
            );


            if (!images || images.length === 0) {
                throw new Error("A IA não conseguiu gerar imagens de variação.");
            }

            if (language !== 'pt') {
                const generatedImageAnalysisSchema = {
                    type: Type.OBJECT,
                    properties: {
                         translation_pt_br: { type: Type.STRING, description: "Uma tradução precisa do texto extraído para o Português do Brasil, com quebras de linha (\n) para manter a formatação original." }
                    },
                    required: ['translation_pt_br']
                };
                
                const imagesWithData = await Promise.all(images.map(async (imgSrc) => {
                    try {
                        const imagePart = { inlineData: { data: imgSrc.split(',')[1], mimeType: 'image/png' } };
                        const translationPrompt = "Extraia todo o texto desta imagem e traduza-o para o Português do Brasil, preservando as quebras de linha originais. Se não houver texto, retorne strings vazias.";
                        
                        const translationResponse = await ai.models.generateContent({
                            model: 'gemini-2.5-flash',
                            contents: { parts: [imagePart, { text: translationPrompt }] },
                            config: {
                                responseMimeType: "application/json",
                                responseSchema: generatedImageAnalysisSchema
                            }
                        });

                        const result = JSON.parse(translationResponse.text);
                        return {
                            src: imgSrc,
                            translation: result.translation_pt_br
                        };
                    } catch (e) {
                        console.error("Falha ao traduzir a imagem gerada:", e);
                        return { src: imgSrc, translation: 'Não foi possível traduzir o texto.' };
                    }
                }));
                setGeneratedImages(imagesWithData);
            } else {
                setGeneratedImages(images.map(src => ({ src, translation: '' })));
            }
        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'Ocorreu um erro desconhecido ao gerar variações.');
        } finally {
            setIsLoading(false);
        }
    };
    
    const exportCreative = (imageUrl: string) => {
        const link = document.createElement('a');
        link.download = `creative-variation-${Date.now()}.png`;
        link.href = imageUrl;
        link.click();
    };

    return (
        <div className="module-container">
            <header className="module-header">
                <h1>Gerador de Variações de Criativos</h1>
                <p>Envie um criativo de referência para análise e gere 5 variações para testes A/B.</p>
            </header>
            <div className="content-wrapper">
                <div className="input-section">
                    <h2>1. Envie seu Criativo</h2>
                    <div className="dropzone">
                        <input type="file" id="file-upload" accept="image/*" onChange={handleFileChange} />
                        <label htmlFor="file-upload" style={{ opacity: isAnalyzing ? 0.5 : 1 }}>
                            <UploadIcon />
                            <span>Clique para selecionar ou arraste uma imagem</span>
                            <small>JPG, PNG, WebP</small>
                        </label>
                    </div>

                    {isAnalyzing && (
                        <div className="analysis-placeholder">
                            <div className="spinner"></div>
                            <p>Analisando criativo...</p>
                        </div>
                    )}
                    
                    {referencePreview && !isAnalyzing && (
                        <div className="analysis-section">
                            <div className="reference-preview">
                                <img src={referencePreview} alt="Criativo de Referência" />
                            </div>
                            {error && <p className="error-message">{error}</p>}
                            {analysis && (
                                <div className="analysis-results">
                                    <h3>2. Análise do Criativo</h3>
                                    {analysis.language !== 'pt' && analysis.translation && (
                                        <div className="analysis-card">
                                            <strong>Tradução ({analysis.language}):</strong>
                                            <p>{analysis.translation}</p>
                                        </div>
                                    )}
                                    <div className="analysis-card">
                                        <strong>Texto Extraído:</strong>
                                        <ul>
                                            <li><strong>Título:</strong> {analysis.text_elements.title}</li>
                                            {analysis.text_elements.points.map((p, i) => <li key={i}>- {p}</li>)}
                                            {analysis.text_elements.secondary_text.map((p, i) => <li key={i}>- {p}</li>)}
                                            <li><strong>CTA:</strong> {analysis.text_elements.cta}</li>
                                        </ul>
                                    </div>
                                    <button onClick={generateVariations} disabled={isLoading}>
                                        {isLoading ? 'Gerando...' : '3. Gerar 5 Variações'}
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="output-section">
                    <h2>Variações Geradas</h2>
                    <div className="output-content">
                        {isLoading && <div className="spinner"></div>}
                        {!isLoading && generatedImages.length === 0 && (
                            <p className="placeholder">Suas variações aparecerão aqui...</p>
                        )}
                        {generatedImages.length > 0 && (
                             <div className="variations-grid">
                                {generatedImages.map((img, index) => (
                                    <div key={index} className="creative-card">
                                        <div className="creative-image-container">
                                            <img src={img.src} alt={`Variação de criativo ${index + 1}`} />
                                            <button className="export-creative-btn" onClick={() => exportCreative(img.src)} title="Exportar Variação">
                                                <ExportIcon />
                                            </button>
                                        </div>
                                        {img.translation && (
                                            <div className="translation-card">
                                                <strong>Tradução:</strong>
                                                <p className="translation-text">{img.translation}</p>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

/**
 * The Batch Video Editor module.
 * Allows users to upload up to 5 videos, apply a master CTA with automatic translation,
 * and export all variations at once.
 */
const BatchVideoEditor = () => {
    const [videoSlots, setVideoSlots] = useState<VideoSlot[]>(
        Array.from({ length: 5 }, (_, i) => ({ id: i, file: null, url: null, thumbnail: null, lang: 'pt' }))
    );
    const [masterCta, setMasterCta] = useState<VideoElement>({
        id: 'master-cta', type: 'text', content: 'Compre Agora!',
        x: 50, y: 85, size: 32,
        color: '#FFFFFF', hasBg: true, bgColor: '#000000',
        fontFamily: "'Anton', sans-serif", fontWeight: 'bold', fontStyle: 'normal', textTransform: 'uppercase',
        hasStroke: false, strokeColor: '#1F2937', strokeWidth: 2, aspectRatio: 1, textAlign: 'center'
    });

    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [exportMessage, setExportMessage] = useState('');
    const [error, setError] = useState<string | null>(null);

    const [draggingElement, setDraggingElement] = useState<{ id: string, initialX: number, initialY: number, mouseInitialX: number, mouseInitialY: number } | null>(null);
    const [isFontPickerOpen, setIsFontPickerOpen] = useState(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const videoWrapperRef = useRef<HTMLDivElement>(null);
    const fontPickerRef = useRef<HTMLDivElement>(null);
    const fileInputRefs = useRef<(HTMLInputElement | null)[]>([]);
    const cancelExportRef = useRef(false);

    const ai = useMemo(() => new GoogleGenAI({ apiKey: process.env.API_KEY }), []);
    const previewVideoUrl = useMemo(() => videoSlots.find(s => s.url)?.url || null, [videoSlots]);
    const hasVideos = useMemo(() => videoSlots.some(s => s.file), [videoSlots]);

    useEffect(() => {
        return () => {
            videoSlots.forEach(slot => {
                if (slot.url) URL.revokeObjectURL(slot.url);
                if (slot.thumbnail) URL.revokeObjectURL(slot.thumbnail);
            });
        };
    }, []);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (draggingElement && videoWrapperRef.current) {
                const rect = videoWrapperRef.current.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;
                updateMasterCta({
                    x: Math.max(0, Math.min(100, x)),
                    y: Math.max(0, Math.min(100, y))
                });
            }
        };
        const handleMouseUp = () => setDraggingElement(null);
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [draggingElement]);

    const generateThumbnail = (videoFile: File): Promise<string> => {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(videoFile);
            video.onloadeddata = () => {
                video.currentTime = 1; // Seek to 1 second
            };
            video.onseeked = () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL('image/jpeg'));
                URL.revokeObjectURL(video.src);
            };
        });
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>, slotId: number) => {
        const file = event.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            const thumbnail = await generateThumbnail(file);
            setVideoSlots(prev => prev.map(slot =>
                slot.id === slotId ? { ...slot, file, url, thumbnail } : slot
            ));
        }
    };

    const removeVideo = (slotId: number) => {
        setVideoSlots(prev => prev.map(slot => {
            if (slot.id === slotId) {
                if (slot.url) URL.revokeObjectURL(slot.url);
                if (slot.thumbnail) URL.revokeObjectURL(slot.thumbnail);
                return { ...slot, file: null, url: null, thumbnail: null };
            }
            return slot;
        }));
    };
    
    const updateSlotLang = (slotId: number, lang: string) => {
        setVideoSlots(prev => prev.map(slot => slot.id === slotId ? { ...slot, lang } : slot));
    };

    const updateMasterCta = (updates: Partial<VideoElement>) => {
        setMasterCta(prev => ({ ...prev, ...updates }));
    };

    const handleElementMouseDown = (e: React.MouseEvent<HTMLDivElement>, id: string) => {
        setDraggingElement({
            id,
            initialX: masterCta.x,
            initialY: masterCta.y,
            mouseInitialX: e.clientX,
            mouseInitialY: e.clientY
        });
        e.stopPropagation();
    };
    
    const handleCancelExport = () => {
        cancelExportRef.current = true;
    };

    const renderAndDownloadVideo = (videoFile: File, ctaElement: VideoElement, fileName: string, onProgress: (p: number) => void): Promise<void> => {
        return new Promise((resolve, reject) => {
            const videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(videoFile);
            let recorder: MediaRecorder | null = null;
            let audioContext: AudioContext | null = null;
            
            const cleanup = () => {
                if(recorder && recorder.state === 'recording') recorder.stop();
                audioContext?.close();
                URL.revokeObjectURL(videoElement.src);
            }

            videoElement.onloadedmetadata = async () => {
                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    cleanup();
                    return reject('Falha ao criar contexto do canvas.');
                }
                
                audioContext = new AudioContext();
                const mixedAudioDestination = audioContext.createMediaStreamDestination();
                
                let hasVideoAudio = false;
                try {
                     if ((videoElement as any).captureStream) {
                        const videoAudioStream = (videoElement as any).captureStream();
                         if (videoAudioStream.getAudioTracks().length > 0) {
                            const videoAudioSource = audioContext.createMediaStreamSource(videoAudioStream);
                            videoAudioSource.connect(mixedAudioDestination);
                            hasVideoAudio = true;
                         }
                     }
                } catch(e) { console.warn("Não foi possível capturar o áudio do vídeo.", e); }
                
                const videoStream = canvas.captureStream(30);
                const combinedStream = new MediaStream([videoStream.getVideoTracks()[0], ...(hasVideoAudio ? mixedAudioDestination.stream.getAudioTracks() : [])]);
                
                const { mimeType, extension } = (() => {
                    const mp4 = 'video/mp4; codecs=avc1.42E01E,mp4a.40.2';
                    if (MediaRecorder.isTypeSupported(mp4)) return { mimeType: mp4, extension: 'mp4' };
                    return { mimeType: 'video/webm; codecs=vp8,opus', extension: 'webm' };
                })();
                
                recorder = new MediaRecorder(combinedStream, { mimeType, videoBitsPerSecond: 8000000 });
                const chunks: Blob[] = [];

                recorder.ondataavailable = e => chunks.push(e.data);
                recorder.onstop = () => {
                    if (cancelExportRef.current) {
                        reject(new Error("Exportação cancelada"));
                        return;
                    }
                    const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${fileName}.${extension}`;
                    a.click();
                    URL.revokeObjectURL(url);
                    cleanup();
                    resolve();
                };
                recorder.onerror = (e) => {
                    console.error("Erro no MediaRecorder:", e)
                    cleanup();
                    reject('Erro durante a gravação do vídeo.');
                }

                videoElement.muted = true;
                recorder.start();
                
                const FPS = 30;
                const totalFrames = Math.ceil(videoElement.duration * FPS);
                
                try {
                    for (let i = 0; i < totalFrames; i++) {
                        if (cancelExportRef.current) {
                             throw new Error("Exportação cancelada pelo usuário.");
                        }

                        const time = i / FPS;
                        const seekPromise = new Promise<void>((res, rej) => {
                            const timeoutId = setTimeout(() => rej(new Error("Timeout de busca de vídeo")), 3000);
                            videoElement.onseeked = () => { clearTimeout(timeoutId); res(); };
                        });
                        videoElement.currentTime = Math.min(time, videoElement.duration);
                        await seekPromise;

                        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

                        const x = (ctaElement.x / 100) * canvas.width;
                        const y = (ctaElement.y / 100) * canvas.height;
                        const scaledFontSize = Math.round(ctaElement.size * (canvas.height / 500));
                        ctx.font = `${ctaElement.fontStyle} ${ctaElement.fontWeight} ${scaledFontSize}px ${ctaElement.fontFamily}`;
                        ctx.textAlign = ctaElement.textAlign;
                        ctx.textBaseline = 'middle';
                        const lines = ctaElement.content.split(/\r?\n/).map(line => ctaElement.textTransform === 'uppercase' ? line.toUpperCase() : line);
                        const lineHeight = scaledFontSize * 1.2;
                        
                        const drawTextLines = (drawFunc: (line: string, y: number) => void) => {
                             const startY = y - ((lines.length - 1) * lineHeight) / 2;
                            lines.forEach((line, index) => {
                                const currentY = startY + (index * lineHeight);
                                drawFunc(line, currentY);
                            });
                        };

                        if (ctaElement.hasBg) {
                            let maxWidth = 0;
                            lines.forEach(line => {
                                const metrics = ctx.measureText(line);
                                if (metrics.width > maxWidth) maxWidth = metrics.width;
                            });
                            const padding = scaledFontSize * 0.2;
                            ctx.fillStyle = ctaElement.bgColor;
                            const totalHeight = (lines.length * lineHeight);
                            const bgY = y - totalHeight / 2;
                            let bgX = x;
                            if (ctx.textAlign === 'center') bgX = x - maxWidth / 2;
                            else if (ctx.textAlign === 'right') bgX = x - maxWidth;
                            ctx.fillRect(bgX - padding, bgY - padding, maxWidth + padding * 2, totalHeight + padding * 1.5);
                        }
                        if (ctaElement.hasStroke) {
                            ctx.strokeStyle = ctaElement.strokeColor;
                            ctx.lineWidth = ctaElement.strokeWidth * (canvas.height / 500);
                            drawTextLines((line, lineY) => ctx.strokeText(line, x, lineY));
                        }
                        ctx.fillStyle = ctaElement.color;
                        drawTextLines((line, lineY) => ctx.fillText(line, x, lineY));

                        onProgress((i / totalFrames) * 100);
                    }
                } catch (err) {
                     cleanup();
                     return reject(err);
                }
                
                if (recorder.state === 'recording') recorder.stop();
            };
            videoElement.onerror = () => {
                cleanup();
                reject('Falha ao carregar o arquivo de vídeo para renderização.');
            };
        });
    };

    const handleBatchExport = async () => {
        const slotsToProcess = videoSlots.filter(s => s.file);
        if (slotsToProcess.length === 0) {
            setError('Adicione pelo menos um vídeo para exportar.');
            return;
        }

        setIsExporting(true);
        setExportProgress(0);
        setError(null);
        cancelExportRef.current = false;

        for (let i = 0; i < slotsToProcess.length; i++) {
             if (cancelExportRef.current) {
                setError("Exportação cancelada.");
                break;
            }
            const slot = slotsToProcess[i];
            const langName = LANGUAGES.find(l => l.code === slot.lang)?.name || slot.lang;
            
            try {
                let translatedContent = masterCta.content;
                if (slot.lang !== 'pt') {
                    const transResponse = await ai.models.generateContent({
                        model: 'gemini-2.5-flash',
                        contents: `Traduza o seguinte texto de forma curta e direta para ${langName} para um botão de call-to-action. Retorne APENAS o texto traduzido:\n\n"${masterCta.content}"`
                    });
                    translatedContent = transResponse.text.trim();
                }

                const ctaForThisVideo: VideoElement = {
                    ...masterCta,
                    content: translatedContent,
                };
                
                await renderAndDownloadVideo(slot.file!, ctaForThisVideo, `video_cta_${slot.lang}_${i + 1}`, (progress) => {
                    const overallProgress = ((i + progress / 100) / slotsToProcess.length) * 100;
                    setExportProgress(overallProgress);
                    setExportMessage(`Renderizando ${i + 1}/${slotsToProcess.length} (${langName}): ${Math.round(progress)}%`);
                });

            } catch (err) {
                console.error(err);
                const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
                if (errorMessage.includes("cancelada")) {
                    setError("Exportação cancelada pelo usuário.");
                } else {
                    setError(`Falha ao processar o vídeo ${i + 1}: ${errorMessage}`);
                }
                break; // Stop on first error or cancellation
            }
        }
        
        if (!error && !cancelExportRef.current) {
            setExportMessage('Exportação concluída!');
        }
        
        setTimeout(() => {
            setIsExporting(false);
            setExportMessage('');
            setError(null);
        }, 3000);
    };

    return (
        <div className="module-container video-editor batch-editor">
            {isExporting && (
                <div className="export-overlay">
                    <div className="export-modal">
                        <h2>Exportando Vídeos</h2>
                        <p>{exportMessage}</p>
                        <div className="progress-bar-container">
                            <div className="progress-bar" style={{ width: `${exportProgress}%` }}></div>
                        </div>
                        <p>{Math.round(exportProgress)}%</p>
                        <button className="cancel-button" onClick={handleCancelExport}>Cancelar</button>
                    </div>
                </div>
            )}
            <header className="module-header">
                <h1>Editor de Anúncios em Lote</h1>
                <p>Envie até 5 vídeos, defina um CTA e exporte todas as variações com tradução automática.</p>
            </header>

            {!hasVideos ? (
                <div className="initial-upload-view">
                    <h2>Comece enviando seus vídeos</h2>
                    <div className="video-slots-grid">
                        {videoSlots.map(slot => (
                            <div key={slot.id} className="video-slot empty">
                                <input type="file" id={`video-upload-${slot.id}`} accept="video/*" onChange={e => handleFileChange(e, slot.id)} />
                                <label htmlFor={`video-upload-${slot.id}`}>
                                    <UploadIcon style={{width: 32, height: 32}}/>
                                    <span>Slot {slot.id + 1}</span>
                                </label>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="batch-editor-layout">
                    <div className="batch-main-content">
                        <div className="video-slots-grid">
                            {videoSlots.map(slot => (
                                <div key={slot.id} className={`video-slot ${slot.file ? 'filled' : 'empty'}`}>
                                    {slot.file ? (
                                        <>
                                            <img src={slot.thumbnail!} alt={`Thumbnail for video ${slot.id + 1}`} className="slot-thumbnail" />
                                            <div className="slot-overlay">
                                                <button onClick={() => removeVideo(slot.id)} className="remove-video-btn"><TrashIcon /></button>
                                                <select value={slot.lang} onChange={e => updateSlotLang(slot.id, e.target.value)}>
                                                    {LANGUAGES.map(lang => <option key={lang.code} value={lang.code}>{lang.name}</option>)}
                                                </select>
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <input type="file" id={`video-upload-${slot.id}`} accept="video/*" onChange={e => handleFileChange(e, slot.id)} />
                                            <label htmlFor={`video-upload-${slot.id}`}>
                                                <UploadIcon style={{width: 32, height: 32}}/>
                                                <span>Slot {slot.id + 1}</span>
                                            </label>
                                        </>
                                    )}
                                </div>
                            ))}
                        </div>
                        <div className="preview-section">
                            <h3>Preview (usando o primeiro vídeo)</h3>
                            <div className="device-preview aspect-916">
                                <div className="video-wrapper" ref={videoWrapperRef}>
                                    {previewVideoUrl ? (
                                        <>
                                            <video key={previewVideoUrl} ref={videoRef} src={previewVideoUrl} controls muted loop />
                                            <div className="elements-overlay">
                                                <div
                                                    className="video-element-wrapper selected"
                                                    style={{
                                                        position: 'absolute', left: `${masterCta.x}%`, top: `${masterCta.y}%`,
                                                        transform: 'translate(-50%, -50%)',
                                                    }}
                                                    onMouseDown={(e) => handleElementMouseDown(e, masterCta.id)}
                                                >
                                                    <span className="video-element-text" style={{
                                                        fontSize: `${masterCta.size}px`, color: masterCta.color, fontFamily: masterCta.fontFamily,
                                                        fontWeight: masterCta.fontWeight, fontStyle: masterCta.fontStyle,
                                                        textTransform: masterCta.textTransform as any, backgroundColor: masterCta.hasBg ? masterCta.bgColor : 'transparent',
                                                        padding: masterCta.hasBg ? '0.2em 0.4em' : '0', borderRadius: '4px', whiteSpace: 'pre-wrap',
                                                        WebkitTextStroke: masterCta.hasStroke ? `${masterCta.strokeWidth}px ${masterCta.strokeColor}` : 'unset',
                                                    }}>
                                                        {masterCta.content}
                                                    </span>
                                                </div>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="placeholder-preview">Carregue um vídeo para ver a pré-visualização</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="batch-controls-section">
                        <h2>Master CTA</h2>
                        <div className="control-group">
                            <div className="style-control" style={{flexWrap: 'wrap', gap: '0.5rem'}}>
                                <label htmlFor="element-text-content" style={{width: '100%'}}>Texto do CTA (em Português)</label>
                                <textarea
                                    id="element-text-content"
                                    value={masterCta.content}
                                    onChange={e => updateMasterCta({ content: e.target.value })}
                                    rows={3}
                                    style={{width: '100%'}}
                                />
                            </div>
                        </div>
                        <div className="control-group">
                            <h3>Estilo do CTA</h3>
                            <div className="style-control">
                                <label>Tamanho</label>
                                <input type="range" min="12" max="100" value={masterCta.size} onChange={e => updateMasterCta({ size: parseInt(e.target.value) })} />
                                <span>{masterCta.size}px</span>
                            </div>
                             <div className="style-control">
                                <label>Estilo</label>
                                <div className="button-group" style={{marginTop: 0, justifyContent: 'flex-end', flexGrow: 1}}>
                                    <button 
                                        className={`style-button ${masterCta.fontWeight === 'bold' ? 'active' : ''}`}
                                        onClick={() => updateMasterCta({ fontWeight: masterCta.fontWeight === 'bold' ? 'normal' : 'bold' })}
                                        title="Negrito"
                                        style={{fontStyle: 'normal'}}
                                    >
                                        B
                                    </button>
                                    <button 
                                        className={`style-button ${masterCta.fontStyle === 'italic' ? 'active' : ''}`}
                                        onClick={() => updateMasterCta({ fontStyle: masterCta.fontStyle === 'italic' ? 'normal' : 'italic' })}
                                        title="Itálico"
                                        style={{fontFamily: 'serif', fontStyle: 'italic'}}
                                    >
                                        I
                                    </button>
                                </div>
                            </div>
                            <div className="style-control">
                                <label>Cor do Texto</label>
                                <input type="color" value={masterCta.color} onChange={e => updateMasterCta({ color: e.target.value })} />
                            </div>
                            <div className="style-control">
                                <label>Fonte</label>
                                <div className="custom-select" ref={fontPickerRef}>
                                    <button className="select-button" style={{ fontFamily: masterCta.fontFamily }} onClick={() => setIsFontPickerOpen(prev => !prev)}>
                                        {FONT_FAMILIES.find(f => f.value === masterCta.fontFamily)?.name || 'Inter'}
                                    </button>
                                    {isFontPickerOpen && (
                                        <ul className="options">
                                            {FONT_FAMILIES.map(font => (
                                                <li key={font.value} style={{ fontFamily: font.value }} onMouseDown={() => { updateMasterCta({ fontFamily: font.value }); setIsFontPickerOpen(false); }}>
                                                    {font.name}
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            </div>
                            <div className="style-control toggle">
                                <label>Fundo</label>
                                <input type="checkbox" id="hasBg" checked={masterCta.hasBg} onChange={e => updateMasterCta({ hasBg: e.target.checked })} />
                                {masterCta.hasBg && <input type="color" value={masterCta.bgColor} onChange={e => updateMasterCta({ bgColor: e.target.value })} />}
                            </div>
                             <div className="style-control toggle">
                                <label>Borda</label>
                                <input type="checkbox" id="hasStroke" checked={masterCta.hasStroke} onChange={e => updateMasterCta({ hasStroke: e.target.checked })} />
                                {masterCta.hasStroke && <input type="color" value={masterCta.strokeColor} onChange={e => updateMasterCta({ strokeColor: e.target.value })} />}
                            </div>
                            {masterCta.hasStroke && (
                                <div className="style-control">
                                    <label>Largura Borda</label>
                                    <input type="range" min="1" max="10" value={masterCta.strokeWidth} onChange={e => updateMasterCta({ strokeWidth: parseInt(e.target.value) })} />
                                    <span>{masterCta.strokeWidth}px</span>
                                </div>
                            )}
                        </div>
                        <div className="control-group export-group">
                             <h3><ExportIcon /> Exportar</h3>
                             <button onClick={handleBatchExport} disabled={isExporting || !hasVideos}>
                                {isExporting ? 'Exportando...' : `Exportar ${videoSlots.filter(s => s.file).length} Vídeo(s)`}
                             </button>
                             {error && <p className="error-message" style={{textAlign: 'center'}}>{error}</p>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};


/**
 * The Video Narrator module.
 * Allows users to upload a video, add subtitles, and generate AI narration.
 */
const VideoNarrator = () => {
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [elements, setElements] = useState<VideoElement[]>([]); 
    const [previewElements, setPreviewElements] = useState<VideoElement[]>([]);
    const [selectedElementId, setSelectedElementId] = useState<string | null>(null);

    const [narrationScript, setNarrationScript] = useState('');
    const [selectedVoice, setSelectedVoice] = useState(VOICES[0].id);
    const [isGenerating, setIsGenerating] = useState(false);
    
    const [selectedLanguages, setSelectedLanguages] = useState(['pt']);
    const [generatedVariations, setGeneratedVariations] = useState<Record<string, { audio: AudioBuffer, subtitles: any[], elements: VideoElement[] }>>({});
    const [selectedVariationLang, setSelectedVariationLang] = useState('pt');

    const [activeAudioBuffer, setActiveAudioBuffer] = useState<AudioBuffer | null>(null);
    const [activeSubtitles, setActiveSubtitles] = useState<{ word: string; startTime: number; endTime: number; index: number; sentenceIndex: number; }[]>([]);
    const [activeSubtitleWordIndex, setActiveSubtitleWordIndex] = useState(-1);
    
    const [subtitlePosition, setSubtitlePosition] = useState({ x: 50, y: 95 });
    
    const [error, setError] = useState<string | null>(null);
    const [originalVideoVolume, setOriginalVideoVolume] = useState(0);
    const [videoStartTime, setVideoStartTime] = useState(0);
    const [narrationEndPadding, setNarrationEndPadding] = useState(3);
    const [startTimeInputText, setStartTimeInputText] = useState("00:00");
    const [durationMode, setDurationMode] = useState<'auto' | 'narration'>('auto');

    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [exportMessage, setExportMessage] = useState('');
    
    const [draggingElement, setDraggingElement] = useState<{id: string, initialX: number, initialY: number, mouseInitialX: number, mouseInitialY: number} | null>(null);
    const [resizingElement, setResizingElement] = useState<{id: string, initialSize: number, mouseInitialX: number, mouseInitialY: number} | null>(null);
    const [isDraggingSubtitle, setIsDraggingSubtitle] = useState(false);
    const subtitleDragStartRef = useRef({ initialX: 0, initialY: 0, mouseX: 0, mouseY: 0 });
    const cancelExportRef = useRef(false);

    const [isFontPickerOpen, setIsFontPickerOpen] = useState(false);
    const fontPickerRef = useRef<HTMLDivElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const previewAudioContextRef = useRef<AudioContext | null>(null);
    const previewAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
    const videoWrapperRef = useRef<HTMLDivElement>(null);
    const imageInputRef = useRef<HTMLInputElement>(null);
    const changeVideoInputRef = useRef<HTMLInputElement>(null);
    const elementImageCache = useRef<Map<string, HTMLImageElement>>(new Map());
    


    const ai = useMemo(() => new GoogleGenAI({ apiKey: process.env.API_KEY }), []);

    const selectedElement = useMemo(() => {
        return elements.find(el => el.id === selectedElementId) || null;
    }, [elements, selectedElementId]);

    useEffect(() => {
        const selectedVariation = generatedVariations[selectedVariationLang];
        if (selectedVariation) {
            setActiveAudioBuffer(selectedVariation.audio);
            setActiveSubtitles(selectedVariation.subtitles);
        } else {
            setActiveAudioBuffer(null);
            setActiveSubtitles([]);
        }
    }, [selectedVariationLang, generatedVariations]);
    
    useEffect(() => {
        const selectedVariation = generatedVariations[selectedVariationLang];
        if (selectedVariation && selectedVariation.elements) {
            setPreviewElements(selectedVariation.elements);
        } else {
            setPreviewElements(elements); // Fallback to original elements if no variation is selected/generated
        }
    }, [selectedVariationLang, generatedVariations, elements]);

    useEffect(() => {
        return () => {
            if (videoUrl) URL.revokeObjectURL(videoUrl);
            previewAudioContextRef.current?.close();
        };
    }, [videoUrl]);
    
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (draggingElement && videoWrapperRef.current) {
                const rect = videoWrapperRef.current.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;
                updateElement(draggingElement.id, { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) });
            }
            if (resizingElement && videoWrapperRef.current) {
                const dx = e.clientX - resizingElement.mouseInitialX;
                const newSize = Math.max(20, resizingElement.initialSize + dx * 2); 
                updateElement(resizingElement.id, { size: newSize });
            }
            if (isDraggingSubtitle && videoWrapperRef.current) {
                const rect = videoWrapperRef.current.getBoundingClientRect();
                const dx = e.clientX - subtitleDragStartRef.current.mouseX;
                const dy = e.clientY - subtitleDragStartRef.current.mouseY;
                const newX = subtitleDragStartRef.current.initialX + (dx / rect.width) * 100;
                const newY = subtitleDragStartRef.current.initialY + (dy / rect.height) * 100;
                setSubtitlePosition({
                    x: Math.max(0, Math.min(100, newX)),
                    y: Math.max(0, Math.min(100, newY))
                });
            }
        };
        const handleMouseUp = () => {
            setDraggingElement(null);
            setResizingElement(null);
            setIsDraggingSubtitle(false);
        };
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [draggingElement, resizingElement, isDraggingSubtitle]);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.volume = originalVideoVolume;
        }
    }, [originalVideoVolume]);
    
    useEffect(() => {
        setStartTimeInputText(new Date(videoStartTime * 1000).toISOString().substr(14, 5));
    }, [videoStartTime]);


    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const isFirstUpload = !videoFile;
    
            // Always update video file and URL
            if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
            }
            setVideoFile(file);
            setVideoUrl(URL.createObjectURL(file));
    
            stopNarrationPreview();
            setError(null);
    
            // If it's the first time a video is uploaded in this session, reset everything to a clean slate.
            if (isFirstUpload) {
                setElements([]);
                setGeneratedVariations({});
                setActiveAudioBuffer(null);
                setActiveSubtitles([]);
                setNarrationScript(''); // Also clear the script
            }
    
            // Always reset preview state for the new video
            setActiveSubtitleWordIndex(-1);
            if (videoRef.current) {
                videoRef.current.currentTime = 0;
                videoRef.current.pause();
            }
        }
        if (event.target) {
            event.target.value = '';
        }
    };

    const stopNarrationPreview = () => {
        if (previewAudioSourceRef.current) {
            previewAudioSourceRef.current.stop();
            previewAudioSourceRef.current.disconnect();
            previewAudioSourceRef.current = null;
        }
    };

    const playNarrationPreview = (offset: number = 0) => {
        if (!activeAudioBuffer) return;
        
        stopNarrationPreview();

        if (!previewAudioContextRef.current || previewAudioContextRef.current.state === 'closed') {
            previewAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        previewAudioContextRef.current.resume();

        const source = previewAudioContextRef.current.createBufferSource();
        source.buffer = activeAudioBuffer;
        source.connect(previewAudioContextRef.current.destination);
        source.start(0, offset);
        previewAudioSourceRef.current = source;
    };

    const handleVideoPlay = () => {
        if (videoRef.current) {
            playNarrationPreview(videoRef.current.currentTime);
        }
    };
    const handleVideoPause = () => {
        stopNarrationPreview();
    };
    const handleVideoSeeking = () => {
        stopNarrationPreview();
    };
    const handleVideoSeeked = () => {
        if (videoRef.current && !videoRef.current.paused) {
            playNarrationPreview(videoRef.current.currentTime);
        }
    };
    
    const handleTimeUpdate = () => {
        if (!videoRef.current || activeSubtitles.length === 0) {
            setActiveSubtitleWordIndex(-1);
            return;
        }
        const currentTime = videoRef.current.currentTime;
        const index = activeSubtitles.findIndex(s => currentTime >= s.startTime && currentTime < s.endTime);
        if (index !== activeSubtitleWordIndex) {
            setActiveSubtitleWordIndex(index);
        }
    };
    
    const handleStartTimeInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        setStartTimeInputText(value);
    
        const parts = value.split(':');
        if (parts.length > 2) return;
    
        const minutes = parseInt(parts[0], 10) || 0;
        const seconds = parseInt(parts[1], 10) || 0;
        
        if (!isNaN(minutes) && !isNaN(seconds)) {
            const totalSeconds = minutes * 60 + seconds;
            if (videoRef.current && totalSeconds <= videoRef.current.duration) {
                setVideoStartTime(totalSeconds);
            }
        }
    };

    const addElement = (type: 'text' | 'image', file?: File) => {
        if (type === 'text') {
            const newElement: VideoElement = {
                id: crypto.randomUUID(), type: 'text', content: 'Seu CTA Aqui',
                x: 50, y: 50, size: 40,
                color: '#FFFFFF', hasBg: true, bgColor: '#000000',
                fontFamily: "'Anton', sans-serif", fontWeight: 'bold', fontStyle: 'normal', textTransform: 'uppercase',
                hasStroke: false, strokeColor: '#000000', strokeWidth: 2, aspectRatio: 1, textAlign: 'center'
            };
            setElements(prev => [...prev, newElement]);
            setSelectedElementId(newElement.id);
        } else if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                const base64Url = reader.result as string;
                const img = new Image();
                img.src = base64Url;
                img.onload = () => {
                    // This is a bit of a hack since VideoElement is text only, but we sneak image properties in
                    const newImageElement: any = {
                        id: crypto.randomUUID(), type: 'image', content: base64Url,
                        x: 50, y: 50, size: 150,
                        aspectRatio: img.width / img.height,
                    };
                    setElements(prev => [...prev, newImageElement]);
                    setSelectedElementId(newImageElement.id);
                };
            };
            reader.readAsDataURL(file);
        }
    };

    const handleMediaUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) addElement('image', file);
        event.target.value = '';
    };

    const removeElement = (id: string) => {
        setElements(prev => prev.filter(el => (el as any).id !== id));
        if (selectedElementId === id) setSelectedElementId(null);
    };
    
    const updateElement = (id: string, updates: Partial<VideoElement>) => {
        setElements(prev => prev.map(el => (el as any).id === id ? { ...el, ...updates } : el));
    };

    const handleElementMouseDown = (e: React.MouseEvent<HTMLDivElement | HTMLImageElement>, id: string) => {
        setSelectedElementId(id);
        const el = elements.find(elem => (elem as any).id === id);
        if (el) {
            setDraggingElement({ id, initialX: el.x, initialY: el.y, mouseInitialX: e.clientX, mouseInitialY: e.clientY });
        }
        e.stopPropagation();
    };

    const handleResizeMouseDown = (e: React.MouseEvent<HTMLDivElement>, id: string) => {
        e.preventDefault();
        e.stopPropagation();
        const el = elements.find(elem => (elem as any).id === id);
        if (el) {
            setSelectedElementId(id);
            // FIX: Corrected property names 'initialX' and 'initialY' to 'mouseInitialX' and 'mouseInitialY' to match the state's type definition.
            setResizingElement({ id, initialSize: el.size, mouseInitialX: e.clientX, mouseInitialY: e.clientY });
        }
    };

    const handleSubtitleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDraggingSubtitle(true);
        subtitleDragStartRef.current = {
            initialX: subtitlePosition.x,
            initialY: subtitlePosition.y,
            mouseX: e.clientX,
            mouseY: e.clientY
        };
    };

    const handleLanguageToggle = (langCode: string) => {
        setSelectedLanguages(prev => {
            if (prev.includes(langCode)) {
                return prev.length > 1 ? prev.filter(c => c !== langCode) : prev;
            }
            return [...prev, langCode];
        });
    };

    const generateAllVariations = async () => {
        if (!narrationScript.trim() || selectedLanguages.length === 0) {
            setError("O roteiro da narração e pelo menos um idioma devem ser selecionados.");
            return;
        }
        setIsGenerating(true);
        setError(null);
        setGeneratedVariations({});
        stopNarrationPreview();

        try {
            const variations: Record<string, { audio: AudioBuffer, subtitles: any[], elements: VideoElement[] }> = {};

            for (const langCode of selectedLanguages) {
                const langName = LANGUAGES.find(l => l.code === langCode)?.name || langCode;
                let translatedScript = narrationScript;

                if (langCode !== 'pt') {
                    const transResponse = await ai.models.generateContent({
                        model: 'gemini-2.5-flash',
                        contents: `Traduza o seguinte texto para ${langName} de forma natural e fluida. Retorne apenas o texto traduzido, sem qualquer outra formatação ou introdução:\n\n"${narrationScript}"`
                    });
                    translatedScript = transResponse.text.trim();
                }

                let translatedElementsForLang: VideoElement[] = [];
                if (langCode === 'pt' || elements.filter(el => el.type === 'text').length === 0) {
                    translatedElementsForLang = JSON.parse(JSON.stringify(elements));
                } else {
                    const textElements = elements.filter(el => el.type === 'text');
                    const translationPromises = textElements.map(el => 
                        ai.models.generateContent({
                            model: 'gemini-2.5-flash',
                            contents: `Traduza o seguinte texto para ${langName} para um CTA em um vídeo. Retorne APENAS o texto traduzido, nada mais:\n\n"${el.content}"`
                        })
                    );
                    const translationResponses = await Promise.all(translationPromises);
                    const translatedTexts = translationResponses.map(res => res.text.trim());
                    
                    let translatedTextIndex = 0;
                    translatedElementsForLang = JSON.parse(JSON.stringify(elements)).map((el: VideoElement) => {
                        if (el.type === 'text') {
                            el.content = translatedTexts[translatedTextIndex] || el.content;
                            translatedTextIndex++;
                        }
                        return el;
                    });
                }

                const sentences = translatedScript.trim().match(/[^.!?\n\r]+[.!?\n\r]*\s*/g) || [translatedScript];
                const sentenceAudioBuffers: AudioBuffer[] = [];
                const allSubtitles: any[] = [];
                let cumulativeDuration = 0;
                let wordGlobalIndex = 0;

                for (const sentence of sentences) {
                    const cleanSentence = sentence.trim();
                    if (!cleanSentence) continue;

                    const audioResponse = await ai.models.generateContent({
                        model: "gemini-2.5-flash-preview-tts",
                        contents: [{ parts: [{ text: cleanSentence }] }],
                        config: {
                            responseModalities: [Modality.AUDIO],
                            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } } },
                        },
                    });
                    const base64Audio = audioResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
                    if (!base64Audio) continue;

                    const audioBytes = decodeBase64(base64Audio);
                    const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
                    const buffer = await decodePcmAudioData(audioBytes, tempAudioContext);
                    tempAudioContext.close();

                    sentenceAudioBuffers.push(buffer);
                    const sentenceDuration = buffer.duration;
                    
                    const wordsWithContext = cleanSentence.split(/\s+/).map(word => ({ word, sentenceIndex: sentenceAudioBuffers.length - 1, index: wordGlobalIndex++ })).filter(w => w.word);
                    const totalCharLength = wordsWithContext.reduce((sum, w) => sum + w.word.length + 1, 0);

                    if (totalCharLength > 0 && sentenceDuration > 0) {
                        let accumulatedChars = 0;
                        const sentenceSubtitles = wordsWithContext.map(w => {
                            const startTime = cumulativeDuration + (accumulatedChars / totalCharLength) * sentenceDuration;
                            accumulatedChars += w.word.length + 1;
                            const endTime = cumulativeDuration + (accumulatedChars / totalCharLength) * sentenceDuration;
                            return { ...w, startTime, endTime };
                        });
                        if (sentenceSubtitles.length > 0) {
                            sentenceSubtitles[sentenceSubtitles.length - 1].endTime = cumulativeDuration + sentenceDuration;
                        }
                        allSubtitles.push(...sentenceSubtitles);
                    }
                    cumulativeDuration += sentenceDuration;
                }
                
                const finalAudioBuffer = await concatenateAudioBuffers(sentenceAudioBuffers, new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 }));
                variations[langCode] = { audio: finalAudioBuffer, subtitles: allSubtitles, elements: translatedElementsForLang };
            }

            setGeneratedVariations(variations);
            setSelectedVariationLang(selectedLanguages[0] || 'pt');

            if (videoRef.current) {
                videoRef.current.currentTime = 0;
                videoRef.current.pause();
            }

        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'Falha ao gerar as variações.');
        } finally {
            setIsGenerating(false);
        }
    };


    const drawKaraokeSubtitles = (
        ctx: CanvasRenderingContext2D,
        currentTime: number,
        canvasWidth: number,
        canvasHeight: number,
        position: { x: number, y: number },
        subtitles: any[]
    ) => {
        if (subtitles.length === 0 || !videoRef.current) return;

        const activeIndex = subtitles.findIndex(s => currentTime >= s.startTime && currentTime < s.endTime);
        
        // FIX: Replace findLastIndex with a reverse loop for wider compatibility.
        let displayIndex = -1;
        for (let i = subtitles.length - 1; i >= 0; i--) {
            if (currentTime >= subtitles[i].startTime) {
                displayIndex = i;
                break;
            }
        }
        
        const narrationEndTime = subtitles.length > 0 ? subtitles[subtitles.length - 1].endTime : 0;
        if (displayIndex === -1 || currentTime >= narrationEndTime) return;
    
        const currentWord = subtitles[displayIndex];
        const currentSentenceIndex = currentWord.sentenceIndex;
        const wordsToDisplay = subtitles.filter(s => s.sentenceIndex === currentSentenceIndex);
    
        if (wordsToDisplay.length === 0) return;
    
        const fontSize = Math.round(32 * (canvasHeight / 720));
        ctx.font = `700 ${fontSize}px 'Poppins', sans-serif`;
        ctx.textBaseline = 'middle';
        ctx.lineJoin = 'round';

        const maxLineWidth = canvasWidth * 0.9;
        const spaceWidth = ctx.measureText(' ').width;
        const lines: {word: string, startTime: number, endTime: number, index: number, sentenceIndex: number}[][] = [];
        let currentLine: {word: string, startTime: number, endTime: number, index: number, sentenceIndex: number}[] = [];
        let currentLineWidth = 0;

        for (const wordObj of wordsToDisplay) {
            const wordWidth = ctx.measureText(wordObj.word).width;
            if (currentLine.length > 0 && currentLineWidth + wordWidth > maxLineWidth) {
                lines.push(currentLine);
                currentLine = [];
                currentLineWidth = 0;
            }
            currentLine.push(wordObj);
            currentLineWidth += wordWidth + spaceWidth;
        }
        if (currentLine.length > 0) {
            lines.push(currentLine);
        }

        if (lines.length === 0) return;
    
        const centerX = (position.x / 100) * canvasWidth;
        const centerY = (position.y / 100) * canvasHeight;
        const lineHeight = fontSize * 1.2;
        const totalBlockHeight = (lines.length * lineHeight);
        const topOfBlockY = centerY - totalBlockHeight / 2;
        
        let measuredMaxLineWidth = 0;
        lines.forEach(line => {
             const lineWidth = line.reduce((sum, {word}) => sum + ctx.measureText(word + ' ').width, 0);
             if (lineWidth > measuredMaxLineWidth) {
                 measuredMaxLineWidth = lineWidth;
             }
        });

        const bgPadding = fontSize * 0.4;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(
            centerX - (measuredMaxLineWidth / 2) - bgPadding,
            topOfBlockY - bgPadding,
            measuredMaxLineWidth + (bgPadding * 2),
            totalBlockHeight + (bgPadding * 2)
        );

        lines.forEach((line, lineIndex) => {
            const lineY = topOfBlockY + (lineHeight / 2) + (lineIndex * lineHeight);
            const totalLineWidth = line.reduce((sum, w) => sum + ctx.measureText(w.word + ' ').width, 0);
            let currentX = centerX - (totalLineWidth / 2);

            ctx.textAlign = 'left';

            line.forEach((wordObj) => {
                const isWordActive = wordObj.index === activeIndex;
                ctx.strokeStyle = 'black';
                ctx.lineWidth = fontSize * 0.15; 
                ctx.strokeText(wordObj.word, currentX, lineY);
                ctx.fillStyle = isWordActive ? '#FBBF24' : '#FFFFFF';
                ctx.fillText(wordObj.word, currentX, lineY);
                currentX += ctx.measureText(wordObj.word + ' ').width;
            });
        });
    };

    const runExportProcess = (langCode: string, onProgress?: (progress: number) => void): Promise<void> => {
        return new Promise(async (resolve, reject) => {
            if (!videoFile) return reject(new Error("Arquivo de vídeo não encontrado."));
            
            const variationData = generatedVariations[langCode];
            if (!variationData || !variationData.audio) return reject(new Error(`Nenhuma narração gerada para ${langCode}.`));
    
            const videoElement = document.createElement('video');
            let recorder: MediaRecorder | null = null;
            let audioContext: AudioContext | null = null;
            
            const cleanup = () => {
                if(recorder && recorder.state === 'recording') recorder.stop();
                videoElement.pause();
                audioContext?.close();
                if (videoElement.src) URL.revokeObjectURL(videoElement.src);
            };

            try {
                const videoReadyPromise = new Promise<void>((res, rej) => {
                    videoElement.onloadedmetadata = () => res();
                    videoElement.onerror = () => rej(new Error('Falha ao carregar o vídeo para renderização.'));
                });
                videoElement.src = URL.createObjectURL(videoFile);
                await videoReadyPromise;
    
                if (!videoElement.duration || !isFinite(videoElement.duration)) throw new Error('A duração do vídeo é inválida.');
        
                if (!onProgress) {
                    setIsExporting(true);
                    setExportProgress(0);
                    setExportMessage('Preparando recursos...');
                    setError(null);
                }
        
                const imageElements = variationData.elements.filter(el => (el as any).type === 'image');
                await Promise.all(imageElements.map(el => new Promise<void>((res, rej) => {
                    if (elementImageCache.current.has((el as any).id)) return res();
                    const img = new Image();
                    img.src = (el as any).content;
                    img.onload = () => { elementImageCache.current.set((el as any).id, img); res(); };
                    img.onerror = () => rej(new Error(`Falha ao carregar imagem.`));
                })));
        
                if (!onProgress) setExportMessage('Mixando faixas de áudio...');
                
                audioContext = new AudioContext();
                const mixedAudioDestination = audioContext.createMediaStreamDestination();
                
                if (originalVideoVolume > 0 && (videoElement as any).captureStream) {
                    try {
                        const videoAudioStream = (videoElement as any).captureStream();
                        if (videoAudioStream.getAudioTracks().length > 0) {
                            const videoAudioSource = audioContext.createMediaStreamSource(videoAudioStream);
                            const videoVolumeNode = audioContext.createGain();
                            videoVolumeNode.gain.value = originalVideoVolume;
                            videoAudioSource.connect(videoVolumeNode);
                            videoVolumeNode.connect(mixedAudioDestination);
                        }
                    } catch (e) { console.warn("Não foi possível capturar o áudio do vídeo.", e); }
                }
                
                const narrationSource = audioContext.createBufferSource();
                narrationSource.buffer = variationData.audio;
                narrationSource.connect(mixedAudioDestination);
                
                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('Falha ao iniciar o canvas para renderização.');
            
                const videoStream = canvas.captureStream(30);
                const combinedStream = new MediaStream([videoStream.getVideoTracks()[0], ...mixedAudioDestination.stream.getAudioTracks()]);
        
                const { mimeType, extension } = (() => {
                    const mp4 = 'video/mp4; codecs=avc1.42E01E,mp4a.40.2';
                    if (MediaRecorder.isTypeSupported(mp4)) return { mimeType: mp4, extension: 'mp4' };
                    return { mimeType: 'video/webm; codecs=vp8,opus', extension: 'webm' };
                })();
                
                recorder = new MediaRecorder(combinedStream, { mimeType, videoBitsPerSecond: 10000000 });
                recorder.onerror = (event) => reject(new Error(`Erro no gravador: ${(event as any).error.message}`));
        
                const chunks: Blob[] = [];
                recorder.ondataavailable = e => chunks.push(e.data);
                recorder.onstop = () => {
                    if (cancelExportRef.current) return reject(new Error("Exportação cancelada pelo usuário."));

                    const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `video_narrado_${langCode}_${Date.now()}.${extension}`;
                    a.click();
                    URL.revokeObjectURL(url);
                    resolve();
                };
        
                const drawOverlays = (currentTime: number) => {
                    variationData.elements.forEach(el_any => {
                        const el = el_any as any;
                        const x = (el.x / 100) * canvas.width;
                        const y = (el.y / 100) * canvas.height;
                        if (el.type === 'text') {
                            const scaledFontSize = Math.round(el.size * (canvas.height / 500));
                            ctx.font = `${el.fontStyle} ${el.fontWeight} ${scaledFontSize}px ${el.fontFamily}`;
                            ctx.textAlign = el.textAlign;
                            ctx.textBaseline = 'middle';
                            const lines = el.content.split(/\r?\n/).map((line: string) => el.textTransform === 'uppercase' ? line.toUpperCase() : line);
                            const lineHeight = scaledFontSize * 1.2;
                            
                            const drawTextLines = (drawFunc: (line: string, y: number) => void) => {
                                const startY = y - ((lines.length - 1) * lineHeight) / 2;
                                lines.forEach((line: string, index: number) => {
                                    const currentY = startY + (index * lineHeight);
                                    drawFunc(line, currentY);
                                });
                            };

                            if (el.hasBg) {
                                let maxWidth = 0;
                                lines.forEach((line: string) => {
                                    const metrics = ctx.measureText(line);
                                    if (metrics.width > maxWidth) maxWidth = metrics.width;
                                });
                                const padding = scaledFontSize * 0.2;
                                const totalHeight = (lines.length * lineHeight);
                                const bgY = y - totalHeight / 2;
                                let bgX = x;
                                if (ctx.textAlign === 'center') bgX = x - maxWidth / 2;
                                else if (ctx.textAlign === 'right') bgX = x - maxWidth;
                
                                ctx.fillStyle = el.bgColor;
                                ctx.fillRect(bgX - padding, bgY - padding, maxWidth + padding * 2, totalHeight + padding * 1.5);
                            }

                            if (el.hasStroke && el.strokeWidth > 0) {
                                ctx.strokeStyle = el.strokeColor;
                                ctx.lineWidth = el.strokeWidth * (canvas.height / 500);
                                drawTextLines((line, lineY) => ctx.strokeText(line, x, lineY));
                            }
                            ctx.fillStyle = el.color;
                            drawTextLines((line, lineY) => ctx.fillText(line, x, lineY));

                        } else if (el.type === 'image') {
                            const img = elementImageCache.current.get(el.id);
                            if (img) {
                                const scaledWidth = (el.size / 500) * canvas.width;
                                const scaledHeight = scaledWidth / el.aspectRatio;
                                ctx.drawImage(img, x - scaledWidth / 2, y - scaledHeight / 2, scaledWidth, scaledHeight);
                            }
                        }
                    });

                    drawKaraokeSubtitles(ctx, currentTime, canvas.width, canvas.height, subtitlePosition, variationData.subtitles);
                }
                
                const narrationDuration = variationData.audio.duration;
                let calculatedDuration = narrationDuration + narrationEndPadding;
                if (durationMode === 'auto') {
                    calculatedDuration = Math.min(calculatedDuration, videoElement.duration - videoStartTime);
                }
                const exportDuration = Math.min(calculatedDuration, videoElement.duration - videoStartTime);

                videoElement.muted = true;

                narrationSource.start(0);
                recorder.start();
        
                const FPS = 30;
                const totalFrames = Math.ceil(exportDuration * FPS);

                for (let i = 0; i < totalFrames; i++) {
                    if (cancelExportRef.current) {
                        throw new Error("Exportação cancelada pelo usuário.");
                    }
            
                    const narrationTime = i / FPS;
                    const videoTime = videoStartTime + narrationTime;
            
                    if (videoTime > videoElement.duration) {
                        break;
                    }
            
                    const seekPromise = new Promise<void>((res, rej) => {
                        const timeoutId = setTimeout(() => rej(new Error("Timeout de busca de vídeo")), 5000);
                        videoElement.onseeked = () => { clearTimeout(timeoutId); res(); };
                        videoElement.onerror = () => rej(new Error("Erro de elemento de vídeo durante a busca"));
                    });
            
                    videoElement.currentTime = videoTime;
                    await seekPromise;
            
                    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                    drawOverlays(narrationTime);
                    onProgress?.((i / totalFrames) * 100);
                }
                
                if (recorder.state === 'recording') {
                    recorder.stop();
                } else {
                    // If the loop finished before recorder was stopped
                    cleanup();
                    resolve();
                }
                
            } catch(err) {
                cleanup();
                reject(err);
            }
        });
    };

    const handleCancelExport = () => {
        cancelExportRef.current = true;
    };

    const handleExportSingle = async () => {
        cancelExportRef.current = false;
        setIsExporting(true);
        setError(null);
        try {
            await runExportProcess(selectedVariationLang, (progress) => {
                setExportProgress(progress);
                setExportMessage(`Renderizando: ${Math.round(progress)}%`);
            });
             setExportMessage('Exportação Concluída!');
        } catch(err) {
            const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
             if (errorMessage.includes("cancelada")) {
                setExportMessage("Exportação cancelada.");
            } else {
                setError(errorMessage);
                setExportMessage("Falha na exportação.");
            }
        } finally {
            setTimeout(() => {
                setIsExporting(false);
                setExportMessage('');
            }, 3000);
        }
    };

    const handleExportAll = async () => {
        setIsExporting(true);
        setExportProgress(0);
        setError(null);
        cancelExportRef.current = false;
        const variationsToExport = Object.keys(generatedVariations);
        
        for (let i = 0; i < variationsToExport.length; i++) {
            if (cancelExportRef.current) {
                setError("Exportação em lote cancelada.");
                break;
            }
            const langCode = variationsToExport[i];
            const langName = LANGUAGES.find(l => l.code === langCode)?.name || langCode;
            
            try {
                await runExportProcess(langCode, (progress) => {
                    const overallProgress = ((i + progress / 100) / variationsToExport.length) * 100;
                    setExportProgress(overallProgress);
                    setExportMessage(`Renderizando ${i + 1}/${variationsToExport.length} (${langName}): ${Math.round(progress)}%`);
                });
            } catch (err) {
                const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
                if (!errorMessage.includes("cancelada")){
                     setError(`Falha ao exportar ${langName}: ${errorMessage}`);
                }
                break;
            }
        }
        
        if (!error && !cancelExportRef.current) {
            setExportMessage('Exportação em lote concluída!');
        } else {
             setExportMessage('Exportação cancelada.');
        }

        setTimeout(() => {
            setIsExporting(false);
            setExportMessage('');
            setError(null);
        }, 4000);
    };


    return (
      <div className="module-container video-editor">
        {isExporting && (
            <div className="export-overlay">
                <div className="export-modal">
                    <h2>Exportando Vídeo</h2>
                    <p>{exportMessage}</p>
                    <div className="progress-bar-container">
                        <div className="progress-bar" style={{ width: `${exportProgress}%` }}></div>
                    </div>
                    <p>{Math.round(exportProgress)}%</p>
                    <button className="cancel-button" onClick={handleCancelExport}>Cancelar</button>
                </div>
            </div>
        )}
        {!videoUrl ? (
          <div className="video-upload-container">
            <h1>Narrador de Vídeo Multilíngue</h1>
            <p>Gere narrações e legendas em múltiplos idiomas para o seu vídeo.</p>
            <div className="dropzone">
              <input type="file" id="video-upload-narrator" accept="video/mp4,video/webm,video/quicktime" onChange={handleFileChange} />
              <label htmlFor="video-upload-narrator">
                <UploadIcon />
                <span>Clique para selecionar ou arraste um vídeo</span>
                <small>MP4, WebM, MOV</small>
              </label>
            </div>
            {error && <p className="error-message">{error}</p>}
          </div>
        ) : (
          <div className="editor-layout">
            <div className="player-section" onClick={() => setSelectedElementId(null)}>
                <div className="device-preview aspect-916">
                    <div className="video-wrapper" ref={videoWrapperRef}>
                        <video key={videoUrl} ref={videoRef} src={videoUrl} controls crossOrigin="anonymous" onTimeUpdate={handleTimeUpdate} onPlay={handleVideoPlay} onPause={handleVideoPause} onSeeking={handleVideoSeeking} onSeeked={handleVideoSeeked} />
                        <div className="elements-overlay">
                            {previewElements.map(el_any => {
                                const el = el_any as any;
                                return (
                                <div
                                    key={el.id}
                                    className={`video-element-wrapper ${selectedElementId === el.id ? 'selected' : ''}`}
                                    style={{
                                        position: 'absolute', left: `${el.x}%`, top: `${el.y}%`, transform: 'translate(-50%, -50%)',
                                        width: el.type === 'image' ? `${el.size}px` : 'auto',
                                        height: el.type === 'image' ? `${el.size / el.aspectRatio}px` : 'auto',
                                        textAlign: 'center',
                                    }}
                                    onMouseDown={(e) => handleElementMouseDown(e, el.id)}
                                >
                                     { (el.type === 'text') && (
                                        <span className="video-element-text" style={{
                                            fontSize: `${el.size}px`, color: el.color, fontFamily: el.fontFamily,
                                            fontWeight: el.fontWeight, fontStyle: el.fontStyle,
                                            textTransform: el.textTransform as any, backgroundColor: el.hasBg ? el.bgColor : 'transparent',
                                            padding: el.hasBg ? '0.2em 0.4em' : '0', borderRadius: '4px', whiteSpace: 'pre-wrap',
                                            WebkitTextStroke: el.hasStroke ? `${el.strokeWidth}px ${el.strokeColor}` : 'unset',
                                        }}>{el.content}</span>
                                    )}
                                    { el.type === 'image' && <img src={el.content} className="video-element-media" alt="Elemento de vídeo" draggable="false" /> }
                                    { selectedElementId === el.id && (el.type === 'image' || el.type === 'text') && <div className="resize-handle" onMouseDown={(e) => handleResizeMouseDown(e, el.id)}></div> }
                                </div>
                            )})}
                             {(() => {
                                const narrationEndTime = activeSubtitles.length > 0 ? activeSubtitles[activeSubtitles.length - 1].endTime : 0;
                                const currentTime = videoRef.current?.currentTime ?? 0;
                                const hasActiveSubtitles = activeSubtitles.length > 0 && currentTime < narrationEndTime;

                                if (!hasActiveSubtitles) {
                                    return null;
                                }

                                return (
                                    <div 
                                        className="subtitle-preview-overlay" 
                                        style={{ left: `${subtitlePosition.x}%`, top: `${subtitlePosition.y}%` }}
                                        onMouseDown={handleSubtitleMouseDown}
                                    >
                                        {(() => {
                                            if (activeSubtitleWordIndex === -1) return null;
                                            const activeWord = activeSubtitles[activeSubtitleWordIndex];
                                            if (!activeWord) return null;
                                            const activeSentenceWords = activeSubtitles.filter(s => s.sentenceIndex === activeWord.sentenceIndex);
                                            if (activeSentenceWords.length === 0) return null;
                                            return (
                                                <p className="subtitle-line">
                                                    {activeSentenceWords.map((wordObj) => (
                                                        <span key={wordObj.index} className={wordObj.index === activeSubtitleWordIndex ? 'active' : ''}>
                                                            {wordObj.word}{' '}
                                                        </span>
                                                    ))}
                                                </p>
                                            );
                                        })()}
                                    </div>
                                );
                            })()}
                        </div>
                    </div>
                </div>
            </div>
            <div className="controls-section">
                <h2>Controles</h2>
                <div className="control-group">
                    <h3><NarrationIcon /> Roteiro Principal (em Português)</h3>
                    <textarea value={narrationScript} onChange={(e) => setNarrationScript(e.target.value)} placeholder="Digite o roteiro da sua narração aqui..." rows={5} style={{width: '100%', resize: 'vertical'}}/>
                    <div className="style-control"><label>Voz</label><select value={selectedVoice} onChange={e => setSelectedVoice(e.target.value)}>{VOICES.map(voice => <option key={voice.id} value={voice.id}>{voice.name}</option>)}</select></div>
                    <div className="control-group">
                        <label className="multi-select-label">Idiomas para Variações</label>
                        <div className="multi-select-container">
                            {LANGUAGES.map(lang => (
                                <button key={lang.code} className={`multi-select-option ${selectedLanguages.includes(lang.code) ? 'selected' : ''}`} onClick={() => handleLanguageToggle(lang.code)}>{lang.name}</button>
                            ))}
                        </div>
                    </div>
                    <button onClick={generateAllVariations} disabled={isGenerating || !narrationScript.trim()}>
                        {isGenerating ? <div className="spinner-small"></div> : null}
                        <span>Gerar Variações</span>
                    </button>
                     {Object.keys(generatedVariations).length > 0 && (
                        <div className="variations-preview-group">
                            <label>Preview da Variação:</label>
                            <div className="variations-buttons">
                                {Object.keys(generatedVariations).map(langCode => (
                                    <button key={langCode} className={selectedVariationLang === langCode ? 'active' : ''} onClick={() => setSelectedVariationLang(langCode)}>
                                        {LANGUAGES.find(l => l.code === langCode)?.name}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                    {error && <p className="error-message">{error}</p>}
                </div>
                 
                <div className="control-group">
                    <h3><ImageIcon /> Camadas (CTA e Imagens)</h3>
                     <div className="add-elements-controls" style={{marginBottom: '1rem', gridTemplateColumns: '1fr 1fr'}}><button onClick={() => addElement('text')}><TextIcon /> <span>Adicionar CTA</span></button><button onClick={() => imageInputRef.current?.click()}><GalleryIcon /> <span>Adicionar Imagem</span></button></div>
                     <input type="file" ref={imageInputRef} onChange={handleMediaUpload} accept="image/png, image/jpeg" style={{display: 'none'}} />
                    <div className="elements-list">
                        {elements.map(el_any => {
                            const el = el_any as any;
                            return (
                            <div key={el.id} className={`element-item ${selectedElementId === el.id ? 'selected' : ''}`} onClick={() => setSelectedElementId(el.id)}>
                                <span>{el.type === 'text' ? 'Texto/CTA' : 'Imagem'}: {el.type === 'text' ? el.content : el.id.substring(0,6)}</span>
                                <button className="delete-btn" onClick={(e) => { e.stopPropagation(); removeElement(el.id); }}><TrashIcon /></button>
                            </div>
                        )})}
                         {elements.length === 0 && <small>Nenhuma camada adicionada.</small>}
                    </div>
                </div>

                {selectedElement && (
                <div className="control-group">
                    <h3>Estilo da Camada</h3>
                    {(selectedElement as any).type === 'text' && (
                        <>
                        <div className="style-control" style={{flexWrap: 'wrap', gap: '0.5rem'}}><label style={{width: '100%'}}>Conteúdo do Texto</label><textarea value={selectedElement.content} onChange={e => updateElement(selectedElementId!, { content: e.target.value })} rows={3} style={{width: '100%'}}/></div>
                        <div className="style-control">
                            <label>Tamanho</label>
                            <input type="range" min={12} max="200" value={selectedElement.size} onChange={e => updateElement(selectedElementId!, { size: parseInt(e.target.value) })} />
                            <span>{selectedElement.size}px</span>
                        </div>
                        <div className="style-control">
                            <label>Estilo</label>
                            <div className="button-group" style={{marginTop: 0, justifyContent: 'flex-end', flexGrow: 1}}>
                                <button 
                                    className={`style-button ${selectedElement.fontWeight === 'bold' ? 'active' : ''}`}
                                    onClick={() => updateElement(selectedElementId!, { fontWeight: selectedElement.fontWeight === 'bold' ? 'normal' : 'bold' })}
                                    title="Negrito"
                                    style={{fontStyle: 'normal'}}
                                >
                                    B
                                </button>
                                <button 
                                    className={`style-button ${selectedElement.fontStyle === 'italic' ? 'active' : ''}`}
                                    onClick={() => updateElement(selectedElementId!, { fontStyle: selectedElement.fontStyle === 'italic' ? 'normal' : 'italic' })}
                                    title="Itálico"
                                    style={{fontFamily: 'serif', fontStyle: 'italic'}}
                                >
                                    I
                                </button>
                            </div>
                        </div>
                        <div className="style-control"><label>Cor</label><input type="color" value={selectedElement.color} onChange={e => updateElement(selectedElementId!, { color: e.target.value })} /></div>
                        <div className="style-control"><label>Fonte</label><div className="custom-select" ref={fontPickerRef}><button className="select-button" style={{ fontFamily: selectedElement.fontFamily }} onClick={() => setIsFontPickerOpen(prev => !prev)}>{FONT_FAMILIES.find(f => f.value === selectedElement.fontFamily)?.name || 'Inter'}</button>{isFontPickerOpen && (<ul className="options">{FONT_FAMILIES.map(font => (<li key={font.value} style={{ fontFamily: font.value }} onMouseDown={() => {updateElement(selectedElementId!, { fontFamily: font.value });setIsFontPickerOpen(false);}}>{font.name}</li>))}</ul>)}</div></div>
                        <div className="style-control toggle">
                            <label>Fundo</label>
                            <input type="checkbox" id="hasBg" checked={selectedElement.hasBg} onChange={e => updateElement(selectedElementId!, { hasBg: e.target.checked })} />
                            {selectedElement.hasBg && <input type="color" value={selectedElement.bgColor} onChange={e => updateElement(selectedElementId!, { bgColor: e.target.value })} />}
                        </div>
                        <div className="style-control toggle">
                            <label>Borda</label>
                            <input type="checkbox" id="hasStroke" checked={selectedElement.hasStroke} onChange={e => updateElement(selectedElementId!, { hasStroke: e.target.checked })} />
                            {selectedElement.hasStroke && <input type="color" value={selectedElement.strokeColor} onChange={e => updateElement(selectedElementId!, { strokeColor: e.target.value })} />}
                        </div>
                        {selectedElement.hasStroke && (
                            <div className="style-control">
                                <label>Largura Borda</label>
                                <input type="range" min="1" max="10" value={selectedElement.strokeWidth} onChange={e => updateElement(selectedElementId!, { strokeWidth: parseInt(e.target.value) })} />
                                <span>{selectedElement.strokeWidth}px</span>
                            </div>
                        )}
                        </>
                    )}
                     {(selectedElement as any).type === 'image' && (
                         <div className="style-control"><label>Tamanho</label><input type="range" min={50} max="500" value={selectedElement.size} onChange={e => updateElement(selectedElementId!, { size: parseInt(e.target.value) })} /><span>{selectedElement.size}px</span></div>
                     )}
                </div>
                )}
                <div className="control-group">
                     <h3><VideoIcon /> Opções de Áudio e Vídeo</h3>
                     <div className="style-control">
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><SpeakerOnIcon style={{width: '20px'}}/> Volume Original</label>
                        <input
                            type="range"
                            min="0" max="1" step="0.01"
                            value={originalVideoVolume}
                            onChange={e => setOriginalVideoVolume(parseFloat(e.target.value))}
                        />
                        <span>{Math.round(originalVideoVolume * 100)}%</span>
                    </div>
                </div>
                 <div className="control-group">
                    <h3><ClockIcon /> Ponto de Início e Duração</h3>
                    <div className="start-time-control">
                        <div className="style-control">
                            <label>Início em:</label>
                             <input
                                type="text"
                                className="time-input"
                                value={startTimeInputText}
                                onChange={handleStartTimeInputChange}
                                onBlur={() => setStartTimeInputText(new Date(videoStartTime * 1000).toISOString().substr(14, 5))}
                                placeholder="MM:SS"
                            />
                        </div>
                         <div className="button-group" style={{marginTop: '0.5rem'}}>
                            <button className="secondary" onClick={() => setVideoStartTime(videoRef.current?.currentTime || 0)}>Usar tempo atual</button>
                            <button className="secondary" onClick={() => setVideoStartTime(0)}>Resetar</button>
                        </div>
                        <div className="duration-options">
                            <label>Duração do Clipe</label>
                            <div className="radio-group">
                                <label>
                                    <input type="radio" value="auto" checked={durationMode === 'auto'} onChange={() => setDurationMode('auto')} />
                                    Automático
                                </label>
                                <label>
                                    <input type="radio" value="narration" checked={durationMode === 'narration'} onChange={() => setDurationMode('narration')} />
                                    Tempo da Narração
                                </label>
                            </div>
                        </div>
                        <div className="style-control" style={{marginTop: '1rem'}}>
                            <label>Padding Final</label>
                            <input
                                type="number"
                                className="time-input"
                                min="0"
                                max="30"
                                value={narrationEndPadding}
                                onChange={e => setNarrationEndPadding(parseInt(e.target.value, 10) || 0)}
                                style={{width: '60px'}}
                            />
                            <span>segundos</span>
                        </div>
                    </div>
                </div>
                <div className="control-group">
                    <h3><ChangeVideoIcon/> Mudar Vídeo</h3>
                    <input type="file" ref={changeVideoInputRef} onChange={handleFileChange} accept="video/mp4,video/webm,video/quicktime" style={{display: 'none'}} />
                    <button onClick={() => changeVideoInputRef.current?.click()}>Trocar Vídeo</button>
                </div>
                 <div className="control-group">
                    <h3><ExportIcon /> Exportar</h3>
                    <div className="export-buttons-container">
                        <button onClick={handleExportSingle} disabled={isExporting || Object.keys(generatedVariations).length === 0}>
                            {isExporting ? 'Exportando...' : 'Exportar Variação Atual'}
                        </button>
                        {Object.keys(generatedVariations).length > 1 && (
                            <button onClick={handleExportAll} disabled={isExporting}>
                                Exportar Todas
                            </button>
                        )}
                    </div>
                </div>
            </div>
          </div>
        )}
      </div>
    );
};


// --- APP ---
// The main application container that handles module switching.

const App = () => {
    const [activeModule, setActiveModule] = useState('narrator');

    return (
        <div className="app-container">
            <aside className="sidebar">
                <div className="sidebar-header">
                    <h2>Mídia IA</h2>
                </div>
                <nav className="sidebar-nav">
                     <button className={activeModule === 'narrator' ? 'active' : ''} onClick={() => setActiveModule('narrator')}>
                        <NarrationIcon />
                        <span>Narrador de Vídeo</span>
                    </button>
                    <button className={activeModule === 'video' ? 'active' : ''} onClick={() => setActiveModule('video')}>
                        <VideoIcon />
                        <span>Editor de Anúncios</span>
                    </button>
                    <button className={activeModule === 'image' ? 'active' : ''} onClick={() => setActiveModule('image')}>
                        <ImageIcon />
                        <span>Gerador de Imagens</span>
                    </button>
                </nav>
            </aside>
            <main className="main-content">
                {activeModule === 'video' ? <BatchVideoEditor /> : activeModule === 'image' ? <ImageCreator /> : <VideoNarrator />}
            </main>
            <style>{STYLES}</style>
        </div>
    );
};

const STYLES = `
    :root {
        --highlight-color: #FBBF24;
    }
    .app-container {
        display: flex;
        width: 100%;
        height: 100%;
        background-color: var(--background);
    }
    .sidebar {
        width: 240px;
        background-color: var(--surface);
        border-right: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        padding: 1rem;
        flex-shrink: 0;
    }
    .sidebar-header h2 {
        font-size: 1.5rem;
        color: var(--text-primary);
    }
    .sidebar-nav {
        margin-top: 2rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .sidebar-nav button {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border: none;
        background-color: transparent;
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s, color 0.2s;
        text-align: left;
    }
    .sidebar-nav button:hover {
        background-color: var(--border);
        color: var(--text-primary);
    }
    .sidebar-nav button.active {
        background-color: var(--primary);
        color: var(--text-primary);
    }
    .main-content {
        flex-grow: 1;
        overflow: auto;
    }
    .module-container {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    .module-header {
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 1rem;
    }
    .module-header h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .module-header p {
        color: var(--text-secondary);
        font-size: 1rem;
    }
    .content-wrapper {
        display: grid;
        grid-template-columns: minmax(400px, 1fr) 2fr;
        gap: 2rem;
        align-items: flex-start;
    }
    .input-section, .output-section {
        background-color: var(--surface);
        padding: 1.5rem;
        border-radius: 8px;
        height: 100%;
    }
    .input-section {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    .dropzone {
        border: 2px dashed var(--border);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
    }
    .dropzone input[type="file"] {
        display: none;
    }
    .dropzone label {
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
    }
    .dropzone label span {
        color: var(--text-primary);
        font-weight: 500;
    }
    textarea, .prompt-group input {
        width: 100%;
        background-color: var(--background);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
        padding: 0.75rem;
        font-family: var(--font-family);
        font-size: 1rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus, .prompt-group input:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px var(--background), 0 0 0 4px var(--primary);
    }
    button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        background-color: var(--primary);
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 1rem;
    }
    .button-group {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .style-button {
        width: 40px;
        height: 32px;
        padding: 0;
        font-size: 1rem;
        font-weight: bold;
        background-color: var(--background);
        border: 1px solid var(--border);
        color: var(--text-secondary);
    }
    .style-button:hover {
        background-color: var(--border);
        color: var(--text-primary);
    }
    .style-button.active {
        background-color: var(--primary);
        color: var(--text-primary);
        border-color: var(--primary);
    }
    button:hover {
        background-color: var(--primary-hover);
    }
    button:disabled {
        background-color: var(--border);
        cursor: not-allowed;
        color: var(--text-secondary);
    }
    button.secondary {
        background-color: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-weight: 500;
    }
    button.secondary:hover {
        background-color: var(--border);
        color: var(--text-primary);
    }

    .error-message {
        color: #F87171;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .output-content {
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: var(--background);
        border-radius: 6px;
        padding: 1rem;
    }
    .placeholder {
        color: var(--text-secondary);
    }
    .spinner {
        border: 4px solid var(--border);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    .spinner-small {
        border: 2px solid var(--border);
        border-top: 2px solid var(--primary);
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .creative-card {
        width: 100%;
        background-color: var(--surface);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        position: relative;
    }
    .creative-image-container {
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
    }
    .creative-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .translation-card {
        background-color: var(--background);
        padding: 0.75rem;
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
    }
    .translation-card strong {
        color: var(--text-primary);
        display: block;
        margin-bottom: 0.25rem;
    }
    .translation-card p.translation-text {
        color: var(--text-secondary);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .creative-card .export-creative-btn {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        z-index: 10;
        width: 36px;
        height: 36px;
        padding: 0;
        border-radius: 50%;
        background-color: rgba(0,0,0,0.5);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        opacity: 0;
        transition: opacity 0.2s;
    }
    .creative-card:hover .export-creative-btn {
        opacity: 1;
    }
    .creative-card .export-creative-btn:hover {
        background-color: rgba(0,0,0,0.7);
    }
    .creative-card .export-creative-btn svg {
        width: 18px;
        height: 18px;
    }
    .analysis-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        min-height: 200px;
        color: var(--text-secondary);
    }
    .analysis-section {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .reference-preview {
        width: 100%;
        max-height: 300px;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .reference-preview img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .analysis-results { display: flex; flex-direction: column; gap: 1rem; }
    .analysis-results h3 { font-size: 1.1rem; }
    .analysis-card {
        background-color: var(--background);
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .analysis-card strong { color: var(--text-primary); display: block; margin-bottom: 0.5rem; }
    .analysis-card p, .analysis-card ul { color: var(--text-secondary); }
    .analysis-card ul { padding-left: 1rem; display: flex; flex-direction: column; gap: 0.25rem; }
    .variations-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        width: 100%;
    }

    /* Video Editor Styles */
    .video-editor { padding: 0; max-width: none; }
    .video-upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        text-align: center;
        padding: 2rem;
    }
    .video-upload-container h1 { font-size: 2rem; margin-bottom: 0.5rem; }
    .video-upload-container p { color: var(--text-secondary); font-size: 1rem; }
    .video-upload-container .dropzone {
        width: 100%;
        max-width: 600px;
        margin-top: 2rem;
    }
    .editor-layout {
        display: grid;
        grid-template-columns: 1fr 320px;
        height: 100%;
        overflow: hidden;
    }
    .player-section {
        background-color: #000;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        position: relative;
        padding: 2rem;
        width: 100%;
    }
    .device-preview {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        height: 100%;
    }
    .device-preview.aspect-916 {
        width: auto;
        max-width: 380px;
        height: 100%;
        max-height: 800px;
        background: #111;
        border-radius: 44px;
        padding: 14px;
        border: 2px solid #444;
        box-shadow: 0 0 30px rgba(0,0,0,0.6);
    }
    .device-preview.aspect-916 .video-wrapper {
        border-radius: 30px;
        overflow: hidden;
        height: 100%;
    }
    .video-wrapper {
        position: relative;
        width: 100%;
        height: 100%;
        aspect-ratio: 9/16;
    }
    .video-wrapper video {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
    }
    .elements-overlay {
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
    }
    .video-element-wrapper {
        position: absolute;
        pointer-events: auto;
        user-select: none;
        cursor: move;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px; /* Provides a larger grab area */
        transform: translate(-50%, -50%);
    }
    .video-element-wrapper.selected .video-element-text,
    .video-element-wrapper.selected .video-element-media {
        outline: 2px solid var(--primary);
        box-shadow: 0 0 10px var(--primary);
    }
    .video-element-text {
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        white-space: pre-wrap;
        line-height: 1.2;
    }
    .video-element-media {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .resize-handle {
        position: absolute; bottom: 0; right: 0;
        width: 16px; height: 16px; background: var(--primary);
        border: 2px solid white; border-radius: 50%;
        cursor: se-resize; transform: translate(50%, 50%);
    }

    .controls-section {
        background-color: var(--surface);
        padding: 1.5rem;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .control-group h3 {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
    }
    .control-group button { width: 100%; }
    .video-controls, .add-elements-controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }
    .style-control {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        gap: 1rem;
    }
    .style-control.toggle input[type="checkbox"] {
        width: 36px; height: 20px; appearance: none;
        background: var(--background); border-radius: 10px;
        position: relative; cursor: pointer;
    }
    .style-control.toggle input[type="checkbox"]::before {
        content: ''; width: 14px; height: 14px; background: var(--text-secondary);
        border-radius: 50%; position: absolute; top: 3px; left: 3px; transition: all 0.2s;
    }
    .style-control.toggle input[type="checkbox"]:checked { background: var(--primary); }
    .style-control.toggle input[type="checkbox"]:checked::before { transform: translateX(16px); background: white; }
    .style-control label { font-size: 0.9rem; flex-shrink: 0; }
    .style-control input[type="range"] { flex-grow: 1; }
    .style-control input[type="color"] {
        width: 28px; height: 28px; border: none; padding: 0;
        background: none; border-radius: 50%; overflow: hidden;
    }
    .style-control input[type="color"]::-webkit-color-swatch-wrapper { padding: 0; }
    .style-control input[type="color"]::-webkit-color-swatch { border: 1px solid var(--border); border-radius: 50%; }
    .style-control select {
        background-color: var(--background); border: 1px solid var(--border); color: var(--text-primary);
        padding: 0.5rem; border-radius: 4px; width: 100%; flex-grow: 1;
        -webkit-appearance: none; appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%239CA3AF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        background-repeat: no-repeat; background-position: right 0.5rem center; background-size: 1em;
    }
    .custom-select { position: relative; width: 100%; flex-grow: 1; }
    .custom-select .select-button {
        background-color: var(--background); border: 1px solid var(--border); color: var(--text-primary);
        padding: 0.5rem; border-radius: 4px; text-align: left; width: 100%;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%239CA3AF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        background-repeat: no-repeat; background-position: right 0.5rem center; background-size: 1em;
    }
     .custom-select .options {
        position: absolute; bottom: 100%; left: 0; right: 0; background-color: var(--surface);
        border: 1px solid var(--border); border-radius: 4px; list-style: none;
        margin: 0 0 0.25rem 0; padding: 0.5rem; max-height: 200px; overflow-y: auto; z-index: 10;
    }
    .custom-select .options li { padding: 0.5rem; cursor: pointer; border-radius: 4px; transition: background-color 0.2s; }
    .custom-select .options li:hover { background-color: var(--primary); }
    .export-overlay {
        position: fixed; inset: 0; background-color: rgba(0,0,0,0.7);
        display: flex; align-items: center; justify-content: center; z-index: 999;
    }
    .export-modal {
        background-color: var(--surface); padding: 2rem; border-radius: 8px;
        width: 90%; max-width: 400px; text-align: center; display: flex; flex-direction: column; gap: 0.5rem;
    }
    .progress-bar-container {
        height: 10px; background-color: var(--background);
        border-radius: 5px; overflow: hidden; margin-top: 1rem;
    }
    .progress-bar { height: 100%; background-color: var(--primary); transition: width 0.2s; }
    .cancel-button {
        margin-top: 1rem;
        background-color: var(--border);
        color: var(--text-secondary);
    }
    .cancel-button:hover {
        background-color: #4B5563;
        color: var(--text-primary);
    }
    .elements-list {
        margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem;
        max-height: 150px; overflow-y: auto; padding-right: 0.5rem;
    }
    .element-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem; background-color: var(--background); border-radius: 4px;
        border: 1px solid var(--border); cursor: pointer; font-size: 0.9rem; transition: border-color 0.2s;
    }
    .element-item:hover { border-color: var(--text-secondary); }
    .element-item.selected { border-color: var(--primary); }
    .element-item span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .delete-btn {
        padding: 0.25rem; width: auto; background-color: transparent; color: var(--text-secondary);
    }
    .delete-btn:hover { background-color: var(--border); color: #F87171; }
    .subtitle-preview-overlay {
        position: absolute; transform: translate(-50%, -50%); display: flex;
        flex-direction: column; align-items: center; justify-content: center; gap: 0.2em;
        text-align: center; pointer-events: auto; cursor: move;
        background-color: rgba(0,0,0,0.6); border-radius: 8px; padding: 0.4em 0.8em; width: 90%;
    }
    .subtitle-line {
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 2.5vmin;
        line-height: 1.2; color: #FFFFFF; -webkit-text-stroke: 1.5px black;
        paint-order: stroke fill; text-align: center;
    }
    .subtitle-line span { transition: color 0.1s linear; }
    .subtitle-line span.active { color: var(--highlight-color); }
    .multi-select-label {
        font-size: 0.9rem; font-weight: 500; color: var(--text-secondary);
        display: block; margin-bottom: 0.5rem; margin-top: 1rem;
    }
    .multi-select-container { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .multi-select-option {
        padding: 0.5rem 1rem; font-size: 0.9rem; background-color: var(--background);
        border: 1px solid var(--border); width: auto;
    }
     .multi-select-option.selected { background-color: var(--primary); border-color: var(--primary); }
    .variations-preview-group { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
    .variations-preview-group label { font-size: 0.9rem; font-weight: 500; color: var(--text-secondary); }
    .variations-buttons { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .variations-buttons button {
        padding: 0.5rem 1rem; font-size: 0.9rem; background-color: var(--background);
        border: 1px solid var(--border); width: auto;
    }
    .variations-buttons button.active { background-color: var(--primary); border-color: var(--primary); }
    .export-buttons-container {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.5rem;
    }
    .export-buttons-container button:nth-child(2) {
        grid-column: 1 / -1; /* Ocupa as duas colunas se houver duas */
    }
    @media (min-width: 640px) {
        .export-buttons-container {
             grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        }
    }
    .start-time-control {
        background-color: var(--background);
        padding: 0.75rem;
        border-radius: 6px;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .time-input {
        background-color: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-primary);
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
        font-size: 0.9rem;
    }
    .start-time-control span {
        font-weight: 500;
        color: var(--text-secondary);
    }
    .start-time-control .button-group {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0;
    }
    .start-time-control .button-group button {
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    .duration-options {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }
    .duration-options > label {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-secondary);
        display: block;
        margin-bottom: 0.75rem;
    }
    .radio-group {
        display: flex;
        gap: 1.5rem;
        justify-content: flex-start;
    }
    .radio-group label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        cursor: pointer;
    }

    /* Batch Editor Specific Styles */
    .batch-editor.module-container { padding: 0; max-width: none; }
    .batch-editor .module-header { padding: 2rem 2rem 1rem; }
    .initial-upload-view {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center; padding: 2rem;
    }
    .batch-editor-layout {
        display: grid; grid-template-columns: 1fr 350px;
        height: calc(100vh - 120px); overflow: hidden;
    }
    .batch-main-content {
        padding: 0 2rem 2rem;
        display: flex; flex-direction: column; gap: 1rem;
        overflow-y: auto;
    }
    .video-slots-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
    }
    .video-slot {
        aspect-ratio: 9/16; border: 2px dashed var(--border); border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        position: relative; overflow: hidden;
    }
    .video-slot.empty label {
        cursor: pointer; display: flex; flex-direction: column;
        align-items: center; gap: 0.5rem; color: var(--text-secondary);
    }
    .video-slot input[type="file"] { display: none; }
    .slot-thumbnail { width: 100%; height: 100%; object-fit: cover; }
    .slot-overlay {
        position: absolute; inset: 0; background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 50%);
        display: flex; flex-direction: column; justify-content: flex-end; padding: 0.5rem;
    }
    .slot-overlay select {
        width: 100%; font-size: 0.8rem; padding: 0.4rem;
    }
    .remove-video-btn {
        position: absolute; top: 0.5rem; right: 0.5rem; width: 28px; height: 28px;
        padding: 0; border-radius: 50%; background: rgba(0,0,0,0.6); color: white;
    }
    .remove-video-btn svg { width: 14px; height: 14px; }
    .preview-section {
        background-color: var(--surface); padding: 1.5rem;
        border-radius: 8px; margin-top: 1rem; flex-grow: 1; display: flex; flex-direction: column;
    }
    .preview-section h3 { margin-bottom: 1rem; }
    .placeholder-preview {
        width: 100%; height: 100%; display: flex; align-items: center;
        justify-content: center; color: var(--text-secondary); background: #000;
        border-radius: 30px;
    }
    .batch-controls-section {
        background-color: var(--surface); padding: 1.5rem;
        overflow-y: auto; display: flex; flex-direction: column;
        border-left: 1px solid var(--border);
    }
    .batch-controls-section .control-group { margin-bottom: 1rem; }
    .batch-controls-section .export-group {
        margin-top: auto; border-top: 1px solid var(--border);
        padding-top: 1.5rem;
    }

`;

// --- ROOT RENDER ---
// This is the entry point of the application.

createRoot(document.getElementById('root')!).render(<App />);