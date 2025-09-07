import React, { useState, useEffect } from 'react';
import { 
  Lightbulb, 
  Copy, 
  ChevronDown, 
  ChevronUp, 
  Eye, 
  EyeOff,
  Star,
  Clock,
  TrendingUp
} from 'lucide-react';
import { 
  extractKeyInsights, 
  formatInsight, 
  categorizeInsights,
  calculateInsightPriority 
} from './KeyInsights.utils';

const KeyInsights = ({
    data = null, 
    maxInsights = 5,
    autoGenerate = true,
    onInsightClick = null,
    className = ''
}) => {
    const [insights, setInsights] = useState([]);
    const [loading, setLoading] = useState(false); // Fixed: removed space in useState
    const [expandedInsight, setExpandedInsight] = useState(null); // Fixed: changed false to null
    const [hiddenInsights, setHiddenInsights] = useState(new Set());
    const [filter, setFilter] = useState('all');
    const [copiedInsight, setCopiedInsight] = useState(null);

    useEffect(() => {
        if (data && autoGenerate) {
            generateInsights();
        }
    }, [data, maxInsights]);

    const generateInsights = async () => {
        setLoading(true);
        try {
            const rawInsights = await extractKeyInsights(data, maxInsights);
            const categorizedInsights = categorizeInsights(rawInsights); // Fixed: variable name
            const prioritizedInsights = categorizedInsights.map(insight => ({ // Fixed: variable name
                ...insight,
                priority: calculateInsightPriority(insight)
            })).sort((a, b) => b.priority - a.priority);

            setInsights(prioritizedInsights);
        } catch (error) {
            console.error('Error generating insights:', error); // Added error to log
            setInsights([]);
        } finally {
            setLoading(false);
        }
    };

    const handleCopyInsight = async (insight) => {
        try {
            const formattedText = formatInsight(insight);
            await navigator.clipboard.writeText(formattedText);
            setCopiedInsight(insight.id);
            setTimeout(() => setCopiedInsight(null), 2000);
        } catch (error) {
            console.error('Failed to copy insight:', error);
        }
    };

    const toggleInsightExpansion = (insightId) => {
        setExpandedInsight(expandedInsight === insightId ? null : insightId); // Fixed: proper comparison
    };

    const toggleInsightVisibility = (insightId) => {
        const newHidden = new Set(hiddenInsights);
        if (newHidden.has(insightId)) {
            newHidden.delete(insightId);
        } else {
            newHidden.add(insightId);
        }
        setHiddenInsights(newHidden);
    };

    const filteredInsights = insights.filter(insight => {
        if (filter === 'all') return !hiddenInsights.has(insight.id); // Fixed: proper comparison and logic
        return insight.category === filter && !hiddenInsights.has(insight.id); // Fixed: proper comparison
    });

    const getCategoryIcon = (category) => {
        switch (category) {
            case 'trend': return <TrendingUp className="w-4 h-4" />;
            case 'important': return <Star className="w-4 h-4" />;
            case 'recent': return <Clock className="w-4 h-4" />;
            default: return <Lightbulb className="w-4 h-4" />;
        }
    };

    const getCategoryColor = (category) => {
        switch (category) {
            case 'trend': return 'text-green-600 bg-green-50 border-green-200';
            case 'important': return 'text-amber-600 bg-amber-50 border-amber-200';
            case 'recent': return 'text-blue-600 bg-blue-50 border-blue-200';
            default: return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    };

    // Add missing return statement and JSX
    if (loading) {
        return (
            <div className={`p-4 ${className}`}>
                <div className="animate-pulse">
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                </div>
            </div>
        );
    }

    return (
        <div className={`space-y-4 ${className}`}>
            {filteredInsights.map((insight) => (
                <div
                    key={insight.id}
                    className={`p-4 border rounded-lg ${getCategoryColor(insight.category)}`}
                >
                    <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-2">
                            {getCategoryIcon(insight.category)}
                            <div>
                                <h3 className="font-medium">{insight.title}</h3>
                                {expandedInsight === insight.id && (
                                    <p className="mt-2 text-sm">{insight.description}</p>
                                )}
                            </div>
                        </div>
                        <div className="flex space-x-2">
                            <button
                                onClick={() => handleCopyInsight(insight)}
                                className="p-1 hover:bg-white/50 rounded"
                                title="Copy insight"
                            >
                                <Copy className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => toggleInsightVisibility(insight.id)}
                                className="p-1 hover:bg-white/50 rounded"
                                title="Hide insight"
                            >
                                <EyeOff className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => toggleInsightExpansion(insight.id)}
                                className="p-1 hover:bg-white/50 rounded"
                                title="Expand insight"
                            >
                                {expandedInsight === insight.id ? (
                                    <ChevronUp className="w-4 h-4" />
                                ) : (
                                    <ChevronDown className="w-4 h-4" />
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default KeyInsights;