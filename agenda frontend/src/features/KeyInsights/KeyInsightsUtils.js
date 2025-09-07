export const generateInsightId = () => {
    return `insight-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Extract key insights from data
 * @param {string|object|array} data - The data to analyze
 * @param {number} maxInsights - Maximum number of insights to return
 * @returns {Promise<array>} Array of insight objects
 */
export const extractKeyInsights = async (data, maxInsights = 5) => {
    if (!data) return [];

    try {
        const textData = typeof data === 'string' ? data : JSON.stringify(data);

        const sentences = textData
            .split(/[.!?]+/)
            .map(s => s.trim())
            .filter(s => s.length > 10);

        if (sentences.length === 0) return [];

        const insights = [];

        const numericalInsights = extractNumericalInsights(sentences);
        insights.push(...numericalInsights);
        
        const keywordInsights = extractKeywordBasedInsights(sentences);
        insights.push(...keywordInsights);

        const sentimentInsights = extractSentimentInsights(sentences);
        insights.push(...sentimentInsights);

        const patternInsights = extractPatternInsights(sentences);
        insights.push(...patternInsights);

        const uniqueInsights = removeDuplicateInsights(insights);
        return uniqueInsights.slice(0, maxInsights);

    } catch (error) {
        console.error('Error extracting insights:', error);
        return generateFallbackInsights(data);
    }
};

const extractNumericalInsights = (sentences) => {
    const numericalPattern = /(\d+(?:\.\d+)?)\s*([%$€£¥]|\w+)/g;
    const insights = [];

    sentences.forEach((sentence, index) => {
        const matches = sentence.match(numericalPattern);
        if (matches && matches.length > 0) {
            const metrics = matches.map(match => match.trim());
            
            insights.push({
                id: generateInsightId(),
                title: extractTitleFromSentence(sentence),
                summary: sentence.length > 100 ? sentence.substring(0, 100) + '...' : sentence,
                details: sentence,
                category: 'trend',
                metrics: metrics,
                confidence: calculateNumericalConfidence(matches),
                source: `sentence-${index}`,
                timestamp: new Date().toISOString()
            });
        }
    });
    
    return insights;
};

const extractKeywordBasedInsights = (sentences) => {
    const importantKeywords = [
        // Business terms
        'revenue', 'profit', 'growth', 'increase', 'decrease', 'improvement',
        'success', 'failure', 'opportunity', 'risk', 'strategy', 'goal',
        
        // Action terms
        'recommend', 'suggest', 'should', 'must', 'need', 'important',
        'critical', 'urgent', 'priority', 'focus',
        
        // Time-related
        'today', 'tomorrow', 'next', 'previous', 'recently', 'upcoming',
        'quarterly', 'monthly', 'yearly', 'deadline',
        
        // Performance terms
        'performance', 'efficiency', 'productivity', 'quality', 'satisfaction',
        'engagement', 'conversion', 'retention'
    ];

    const insights = [];

    sentences.forEach((sentence, index) => {
        const lowerSentence = sentence.toLowerCase();
        const matchedKeywords = importantKeywords.filter(keyword => 
            lowerSentence.includes(keyword.toLowerCase())
        );

        if (matchedKeywords.length >= 2) {
            insights.push({
                id: generateInsightId(),
                title: extractTitleFromSentence(sentence),
                summary: sentence.length > 120 ? sentence.substring(0, 120) + '...' : sentence,
                details: sentence,
                category: 'important',
                keywords: matchedKeywords,
                confidence: Math.min(matchedKeywords.length * 0.2, 1),
                source: `sentence-${index}`,
                timestamp: new Date().toISOString()
            });
        }
    });
    
    return insights;
};

const extractSentimentInsights = (sentences) => {
    const positiveWords = ['excellent', 'great', 'amazing', 'successful', 'improved', 'better', 'increased', 'positive'];
    const negativeWords = ['poor', 'bad', 'terrible', 'failed', 'decreased', 'worse', 'negative', 'problem'];
    
    const insights = [];

    sentences.forEach((sentence, index) => {
        const lowerSentence = sentence.toLowerCase();
        const positiveCount = positiveWords.filter(word => lowerSentence.includes(word)).length;
        const negativeCount = negativeWords.filter(word => lowerSentence.includes(word)).length;

        if (positiveCount > 0 || negativeCount > 0) {
            const sentiment = positiveCount > negativeCount ? 'positive' : 'negative';
            
            insights.push({
                id: generateInsightId(),
                title: extractTitleFromSentence(sentence),
                summary: sentence.length > 100 ? sentence.substring(0, 100) + '...' : sentence,
                details: sentence,
                category: 'recent',
                sentiment: sentiment,
                sentimentScore: (positiveCount - negativeCount) / (positiveCount + negativeCount),
                confidence: Math.min((positiveCount + negativeCount) * 0.3, 1),
                source: `sentence-${index}`,
                timestamp: new Date().toISOString()
            });
        }
    });

    return insights;
};

const extractPatternInsights = (sentences) => {
    const insights = [];

    sentences.forEach((sentence, index) => {
        // Look for list patterns
        if (sentence.match(/\d+[.)]/g) || sentence.includes(',') && sentence.split(',').length >= 3) {
            insights.push({
                id: generateInsightId(),
                title: 'List or Enumeration Detected',
                summary: sentence.length > 100 ? sentence.substring(0, 100) + '...' : sentence,
                details: sentence,
                category: 'general',
                pattern: 'list',
                confidence: 0.6,
                source: `sentence-${index}`,
                timestamp: new Date().toISOString()
            });
        }

        // Look for comparison patterns
        if (sentence.includes('compared to') || sentence.includes('vs') || sentence.includes('versus')) {
            insights.push({
                id: generateInsightId(),
                title: 'Comparison Analysis',
                summary: sentence.length > 100 ? sentence.substring(0, 100) + '...' : sentence,
                details: sentence,
                category: 'trend',
                pattern: 'comparison',
                confidence: 0.7,
                source: `sentence-${index}`,
                timestamp: new Date().toISOString()
            });
        }
    });

    return insights;
};

const calculateNumericalConfidence = (matches) => {
    if (!matches) return 0;
    
    let score = 0;
    matches.forEach(match => {
        if (match.includes('%')) score += 0.3;
        if (match.includes('$') || match.includes('€') || match.includes('£')) score += 0.3;
        if (/\d+\.\d+/.test(match)) score += 0.2; // Decimal numbers are more precise
        if (parseInt(match) > 100) score += 0.1; // Larger numbers might be more significant
    });

    return Math.min(score, 1);
};

const extractTitleFromSentence = (sentence) => {
    const words = sentence.split(' ').slice(0, 8);
    let title = words.join(' ');
    
    // Clean up the title
    title = title.replace(/[.!?]+$/, '');
    if (title.length > 50) {
        title = title.substring(0, 50) + '...';
    }
    
    return title || 'Key Insight';
};

const removeDuplicateInsights = (insights) => {
    const unique = [];
    const seen = new Set();

    insights.forEach(insight => {
        const key = insight.summary.toLowerCase().replace(/[^\w\s]/g, '').substring(0, 50);
        if (!seen.has(key)) {
            seen.add(key);
            unique.push(insight);
        }
    });

    return unique;
};

const generateFallbackInsights = (data) => {
    const dataStr = typeof data === 'string' ? data : JSON.stringify(data);
    
    return [{
        id: generateInsightId(),
        title: 'Data Overview',
        summary: `Processed ${dataStr.length} characters of data`,
        details: 'Unable to extract specific insights from the provided data format.',
        category: 'general',
        confidence: 0.1,
        source: 'fallback',
        timestamp: new Date().toISOString()
    }];
};

/**
 * Categorize insights based on their content and properties
 * @param {array} insights - Array of insight objects
 * @returns {array} Categorized insights
 */
export const categorizeInsights = (insights) => {
    return insights.map(insight => {
        if (!insight.category || insight.category === 'general') {
            if (insight.metrics || insight.pattern === 'comparison') {
                insight.category = 'trend';
            } else if (insight.confidence > 0.7 || insight.keywords?.length > 2) {
                insight.category = 'important';
            } else if (insight.sentiment) {
                insight.category = 'recent';
            }
        }

        return insight;
    });
};

/**
 * Calculate priority score for insights
 * @param {object} insight - Insight object
 * @returns {number} Priority score (1-5)
 */
export const calculateInsightPriority = (insight) => {
    let score = 1;

    // Base confidence boost
    score += (insight.confidence || 0) * 2;

    // Category-based scoring
    switch (insight.category) {
        case 'important': score += 2; break;
        case 'trend': score += 1.5; break;
        case 'recent': score += 1; break;
        default: score += 0.5;
    }

    // Content-based scoring
    if (insight.metrics?.length > 0) score += 1;
    if (insight.keywords?.length > 2) score += 0.5;
    if (insight.sentiment === 'positive' || insight.sentiment === 'negative') score += 0.5;

    // Length penalty (very long insights might be less focused)
    if (insight.summary.length > 200) score -= 0.5;

    return Math.min(Math.max(Math.round(score), 1), 5);
};

/**
 * Format insight for copying or sharing
 * @param {object} insight - Insight object
 * @returns {string} Formatted insight text
 */
export const formatInsight = (insight) => {
    let formatted = `📋 ${insight.title}\n\n`;
    formatted += `${insight.summary}\n`;
    
    if (insight.details && insight.details !== insight.summary) {
        formatted += `\n📝 Details:\n${insight.details}\n`;
    }

    if (insight.metrics?.length > 0) {
        formatted += `\n📊 Metrics: ${insight.metrics.join(', ')}\n`;
    }

    if (insight.keywords?.length > 0) {
        formatted += `\n🔑 Keywords: ${insight.keywords.join(', ')}\n`;
    }

    formatted += `\n🏷️ Category: ${insight.category}`;
    formatted += `\n⭐ Priority: ${insight.priority || 1}/5`;
    formatted += `\n📅 Generated: ${new Date(insight.timestamp).toLocaleString()}`;

    return formatted;
};

/**
 * Filter insights by various criteria
 * @param {array} insights - Array of insights
 * @param {object} filters - Filter criteria
 * @returns {array} Filtered insights
 */
export const filterInsights = (insights, filters = {}) => {
    return insights.filter(insight => {
        // Category filter
        if (filters.category && insight.category !== filters.category) return false;
        
        // Minimum confidence filter
        if (filters.minConfidence && (insight.confidence || 0) < filters.minConfidence) return false;
        
        // Priority filter
        if (filters.minPriority && (insight.priority || 1) < filters.minPriority) return false;
        
        // Text search filter
        if (filters.searchText) {
            const search = filters.searchText.toLowerCase();
            const searchableText = `${insight.title} ${insight.summary} ${insight.details || ''}`.toLowerCase();
            if (!searchableText.includes(search)) return false;
        }
        
        // Date range filter
        if (filters.startDate) {
            const insightDate = new Date(insight.timestamp);
            if (insightDate < new Date(filters.startDate)) return false;
        }
        
        if (filters.endDate) {
            const insightDate = new Date(insight.timestamp);
            if (insightDate > new Date(filters.endDate)) return false;
        }

        return true;
    });
};

/**
 * Merge similar insights to reduce redundancy
 * @param {array} insights - Array of insights
 * @returns {array} Merged insights
 */
export const mergeSimilarInsights = (insights) => {
    const merged = [];
    const processed = new Set();

    insights.forEach(insight => {
        if (processed.has(insight.id)) return;

        const similar = insights.filter(other => {
            if (processed.has(other.id) || other.id === insight.id) return false;
            return calculateSimilarity(insight, other) > 0.7;
        });

        if (similar.length > 0) {
            const mergedInsight = {
                ...insight,
                id: generateInsightId(),
                title: `${insight.title} (+ ${similar.length} related)`,
                summary: insight.summary,
                details: [insight.details, ...similar.map(s => s.details)].filter(Boolean).join('\n\n'),
                metrics: [...(insight.metrics || []), ...similar.flatMap(s => s.metrics || [])],
                keywords: [...(insight.keywords || []), ...similar.flatMap(s => s.keywords || [])],
                confidence: Math.max(insight.confidence || 0, ...similar.map(s => s.confidence || 0)),
                relatedInsights: similar.map(s => s.id)
            };

            merged.push(mergedInsight);
            processed.add(insight.id);
            similar.forEach(s => processed.add(s.id));
        } else {
            merged.push(insight);
            processed.add(insight.id);
        }
    });

    return merged;
};

/**
 * Calculate similarity between two insights
 * @param {object} insight1 - First insight
 * @param {object} insight2 - Second insight
 * @returns {number} Similarity score (0-1)
 */
const calculateSimilarity = (insight1, insight2) => {
    const text1 = `${insight1.title} ${insight1.summary}`.toLowerCase();
    const text2 = `${insight2.title} ${insight2.summary}`.toLowerCase();
    
    const words1 = new Set(text1.split(/\s+/));
    const words2 = new Set(text2.split(/\s+/));
    
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
};