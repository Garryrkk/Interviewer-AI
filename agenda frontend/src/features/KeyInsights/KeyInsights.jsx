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

import { KeyInsights } from '../../services/aiService';

const KeyInsights = ({
    data = null, 
    maxInsights = 5,
    autoGenerate = true,
    onInsightClick = null,
    className = ''
}) => {
    const [insights, setInsights] = useState([]);
    const [loading, setLoading] = useState(false);
    const [expandedInsight, setExpandedInsight] = useState(null);
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
            const categorizedInsights = categorizeInsights(rawInsights);
            const prioritizedInsights = categorizedInsights.map(insight => ({
                ...insight,
                priority: calculateInsightPriority(insight)
            })).sort((a, b) => b.priority - a.priority);

            setInsights(prioritizedInsights);
        } catch (error) {
            console.error('Error generating insights:', error);
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
        setExpandedInsight(expandedInsight === insightId ? null : insightId);
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
        if (filter === 'all') return !hiddenInsights.has(insight.id);
        return insight.category === filter && !hiddenInsights.has(insight.id);
    });

    const getCategoryIcon = (category) => {
        switch (category) {
            case 'trend': return <TrendingUp className="w-5 h-5" />;
            case 'important': return <Star className="w-5 h-5" />;
            case 'recent': return <Clock className="w-5 h-5" />;
            default: return <Lightbulb className="w-5 h-5" />;
        }
    };

    const getCategoryColor = (category) => {
        switch (category) {
            case 'trend': return 'bg-green-600/20 border-green-600/30 text-green-400';
            case 'important': return 'bg-amber-600/20 border-amber-600/30 text-amber-400';
            case 'recent': return 'bg-blue-600/20 border-blue-600/30 text-blue-400';
            default: return 'bg-slate-700/50 border-slate-600 text-slate-300';
        }
    };

    const getCategoryBadgeColor = (category) => {
        switch (category) {
            case 'trend': return 'bg-green-600 text-white';
            case 'important': return 'bg-amber-600 text-white';
            case 'recent': return 'bg-blue-600 text-white';
            default: return 'bg-slate-600 text-slate-300';
        }
    };

    if (loading) {
        return (
            <div className={`bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700 ${className}`}>
                <div className="animate-pulse space-y-4">
                    <div className="flex items-center space-x-3">
                        <div className="w-6 h-6 bg-slate-700 rounded"></div>
                        <div className="h-6 bg-slate-700 rounded w-1/3"></div>
                    </div>
                    <div className="space-y-3">
                        <div className="h-4 bg-slate-700 rounded w-3/4"></div>
                        <div className="h-4 bg-slate-700 rounded w-1/2"></div>
                        <div className="h-4 bg-slate-700 rounded w-5/6"></div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className={`space-y-4 ${className}`}>
            {/* Filter Bar */}
            {insights.length > 0 && (
                <div className="bg-slate-800/30 backdrop-blur p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-slate-200 flex items-center space-x-2">
                            <Lightbulb className="w-5 h-5 text-red-400" />
                            <span>Key Insights</span>
                        </h3>
                        <div className="flex space-x-2">
                            {['all', 'trend', 'important', 'recent'].map((filterType) => (
                                <button
                                    key={filterType}
                                    onClick={() => setFilter(filterType)}
                                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                                        filter === filterType
                                            ? 'bg-red-600 text-white'
                                            : 'bg-slate-700/50 text-slate-300 hover:bg-slate-600/50'
                                    }`}
                                >
                                    {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Insights List */}
            <div className="space-y-3">
                {filteredInsights.map((insight) => (
                    <div
                        key={insight.id}
                        className={`bg-slate-800/50 backdrop-blur p-6 rounded-xl border transition-all duration-300 ${getCategoryColor(insight.category)}`}
                        onClick={() => onInsightClick && onInsightClick(insight)}
                    >
                        <div className="flex items-start justify-between">
                            <div className="flex items-start space-x-3 flex-1">
                                <div className={`p-2 rounded-lg ${getCategoryBadgeColor(insight.category)}`}>
                                    {getCategoryIcon(insight.category)}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center space-x-2 mb-2">
                                        <h4 className="font-semibold text-slate-200 text-lg">
                                            {insight.title}
                                        </h4>
                                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryBadgeColor(insight.category)}`}>
                                            {insight.category}
                                        </span>
                                        {insight.priority > 80 && (
                                            <span className="px-2 py-1 bg-red-600 text-white text-xs font-medium rounded-full">
                                                High Priority
                                            </span>
                                        )}
                                    </div>
                                    
                                    {/* Always show a preview */}
                                    <p className="text-slate-300 text-sm mb-3 leading-relaxed">
                                        {expandedInsight === insight.id 
                                            ? insight.description 
                                            : `${insight.description?.substring(0, 120)}${insight.description?.length > 120 ? '...' : ''}`
                                        }
                                    </p>

                                    {/* Metadata */}
                                    <div className="flex items-center space-x-4 text-xs text-slate-400">
                                        <span>Priority: {insight.priority}%</span>
                                        {insight.confidence && (
                                            <span>Confidence: {insight.confidence}%</span>
                                        )}
                                        {insight.timestamp && (
                                            <span>{new Date(insight.timestamp).toLocaleTimeString()}</span>
                                        )}
                                    </div>
                                </div>
                            </div>
                            
                            {/* Action Buttons */}
                            <div className="flex items-center space-x-1 ml-4">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleCopyInsight(insight);
                                    }}
                                    className={`p-2 rounded-lg transition-all duration-200 ${
                                        copiedInsight === insight.id
                                            ? 'bg-green-600 text-white'
                                            : 'hover:bg-slate-700/70 text-slate-400 hover:text-slate-200'
                                    }`}
                                    title={copiedInsight === insight.id ? 'Copied!' : 'Copy insight'}
                                >
                                    <Copy className="w-4 h-4" />
                                </button>
                                
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        toggleInsightVisibility(insight.id);
                                    }}
                                    className="p-2 hover:bg-slate-700/70 text-slate-400 hover:text-slate-200 rounded-lg transition-all duration-200"
                                    title="Hide insight"
                                >
                                    <EyeOff className="w-4 h-4" />
                                </button>
                                
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        toggleInsightExpansion(insight.id);
                                    }}
                                    className="p-2 hover:bg-slate-700/70 text-slate-400 hover:text-slate-200 rounded-lg transition-all duration-200"
                                    title={expandedInsight === insight.id ? 'Collapse' : 'Expand'}
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

            {/* Empty State */}
            {filteredInsights.length === 0 && !loading && (
                <div className="bg-slate-800/30 backdrop-blur p-8 rounded-xl border border-slate-700/50 text-center">
                    <Lightbulb className="w-16 h-16 text-slate-500 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-slate-300 mb-2">No Insights Available</h3>
                    <p className="text-slate-400 mb-4">
                        {insights.length === 0 
                            ? 'Generate insights from your data to see key points and analysis here.'
                            : 'No insights match the current filter. Try selecting a different category.'
                        }
                    </p>
                    {insights.length === 0 && (
                        <button
                            onClick={generateInsights}
                            className="bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition-colors font-medium"
                        >
                            Generate Insights
                        </button>
                    )}
                </div>
            )}

            {/* Hidden Insights Counter */}
            {hiddenInsights.size > 0 && (
                <div className="bg-slate-800/30 backdrop-blur p-3 rounded-lg border border-slate-700/50 text-center">
                    <p className="text-slate-400 text-sm">
                        {hiddenInsights.size} insight{hiddenInsights.size !== 1 ? 's' : ''} hidden
                    </p>
                    <button
                        onClick={() => setHiddenInsights(new Set())}
                        className="text-slate-300 hover:text-white text-xs underline ml-2"
                    >
                        Show All
                    </button>
                </div>
            )}
        </div>
    );
};

export default KeyInsights;