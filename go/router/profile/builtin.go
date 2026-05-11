package profile

func RegisterBuiltins(r *Registry) {
	r.Register(Profile{
		Name:            "quiet",
		Tools:           []string{"clear_instruction", "get_instruction", "list_memories", "save_memory", "set_instruction"},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a quiet, helpful IRC assistant. Speak only when addressed.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "chat",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url", "generate_image",
			"get_instruction", "list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a conversational IRC assistant.",
		AllowAmbient:    true,
	})
	r.Register(Profile{
		Name: "scene",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are participating in an ongoing narrative scene. Stay in character.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "loom",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "loom_propose", "loom_seed", "loom_yield",
			"save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are one of several voices weaving a collaborative narrative.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "admin",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url",
			"generate_code", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        5,
		FrameworkPrompt: "You are the assistant. The user has admin privileges.",
		AllowAmbient:    false,
	})
}
