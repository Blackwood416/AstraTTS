using System.Collections.Concurrent;

namespace AstraTTS.Web.Services
{
    public class StreamSession
    {
        public string SessionId { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
        public string? AvatarId { get; set; }
        public string? ReferenceId { get; set; }
        public float Speed { get; set; }
        public float NoiseScale { get; set; }
        public float Temperature { get; set; }
        public int TopK { get; set; }
        public int StreamingChunkSize { get; set; }
        public int StreamingChunkTokens { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool IsCompleted { get; set; }
        public CancellationTokenSource CancellationTokenSource { get; set; } = new();
    }

    public class StreamSessionManager
    {
        private readonly ConcurrentDictionary<string, StreamSession> _sessions = new();
        private readonly ConcurrentDictionary<string, IAsyncEnumerable<float[]>> _streamData = new();

        public string CreateSession(string text, string? avatarId, string? referenceId,
            float speed, float noiseScale, float temperature, int topK,
            int streamingChunkSize, int streamingChunkTokens)
        {
            var sessionId = Guid.NewGuid().ToString("N");
            var session = new StreamSession
            {
                SessionId = sessionId,
                Text = text,
                AvatarId = avatarId,
                ReferenceId = referenceId,
                Speed = speed,
                NoiseScale = noiseScale,
                Temperature = temperature,
                TopK = topK,
                StreamingChunkSize = streamingChunkSize,
                StreamingChunkTokens = streamingChunkTokens,
                CreatedAt = DateTime.UtcNow,
                IsCompleted = false
            };

            _sessions.TryAdd(sessionId, session);
            return sessionId;
        }

        public StreamSession? GetSession(string sessionId)
        {
            _sessions.TryGetValue(sessionId, out var session);
            return session;
        }

        public void SetStreamData(string sessionId, IAsyncEnumerable<float[]> streamData)
        {
            _streamData.TryAdd(sessionId, streamData);
        }

        public IAsyncEnumerable<float[]>? GetStreamData(string sessionId)
        {
            _streamData.TryGetValue(sessionId, out var streamData);
            return streamData;
        }

        public void CompleteSession(string sessionId)
        {
            if (_sessions.TryGetValue(sessionId, out var session))
            {
                session.IsCompleted = true;
            }
        }

        public void CancelSession(string sessionId)
        {
            if (_sessions.TryGetValue(sessionId, out var session))
            {
                session.CancellationTokenSource.Cancel();
            }
        }

        public void RemoveSession(string sessionId)
        {
            if (_sessions.TryRemove(sessionId, out var session))
            {
                session.CancellationTokenSource.Dispose();
            }
            _streamData.TryRemove(sessionId, out _);
        }

        public void CleanupOldSessions(TimeSpan maxAge)
        {
            var cutoff = DateTime.UtcNow - maxAge;
            var expiredSessions = _sessions.Where(s => s.Value.CreatedAt < cutoff && s.Value.IsCompleted).ToList();

            foreach (var expired in expiredSessions)
            {
                RemoveSession(expired.Key);
            }
        }

        public IEnumerable<StreamSession> GetAllSessions()
        {
            return _sessions.Values;
        }
    }
}
