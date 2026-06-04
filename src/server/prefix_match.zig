//! Small prefix-cache matching helpers for OpenAI-style prompt reuse.

pub fn commonLen(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

pub fn boundedStaticReuseLen(static_ids: []const u32, prompt: []const u32, max_cache_tokens: usize, max_total_tokens: usize) usize {
    return @min(commonLen(static_ids, prompt), max_cache_tokens, max_total_tokens);
}
