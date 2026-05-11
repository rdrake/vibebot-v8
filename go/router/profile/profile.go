// Package profile defines the v9 engagement profiles and the lookup registry.
package profile

import "sort"

type Profile struct {
	Name            string
	Tools           []string
	Model           string
	MaxIters        int
	FrameworkPrompt string
	AllowAmbient    bool
}

type Registry struct {
	m map[string]Profile
}

func NewRegistry() *Registry { return &Registry{m: map[string]Profile{}} }

// Register inserts (or overwrites) a profile. Tool list is sorted on insert.
func (r *Registry) Register(p Profile) {
	tools := append([]string(nil), p.Tools...)
	sort.Strings(tools)
	p.Tools = tools
	r.m[p.Name] = p
}

// Get returns a deep copy of the profile, so callers cannot mutate the
// registry's Tools slice through the returned value.
func (r *Registry) Get(name string) (Profile, bool) {
	p, ok := r.m[name]
	if !ok {
		return Profile{}, false
	}
	out := p
	out.Tools = append([]string(nil), p.Tools...)
	return out, true
}
