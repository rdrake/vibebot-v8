package exec

// chunkAt splits s into pieces of at most n bytes. n<=0 returns [s].
// Byte-naive on purpose; sub-project A is responsible for wire-level
// validation (UTF-8 safety, IRC line length).
func chunkAt(s string, n int) []string {
	if n <= 0 || len(s) <= n {
		return []string{s}
	}
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if len(s) > 0 {
		out = append(out, s)
	}
	return out
}
