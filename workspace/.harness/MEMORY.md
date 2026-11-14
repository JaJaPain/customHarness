# Godot 4 Project Durable Memory (.harness/MEMORY.md)
<!-- Static pinned rules & architectural facts extracted before context compaction -->

- **Engine Target**: Godot 4.3+ (`Forward Plus`), GDScript 2.0 (`@export`, `@onready`, `CharacterBody3D`, `Node3D`, `await`, `.instantiate()`).
- **3D Coordinate & Physics Convention**: glTF `+Y` Up, `1.0m` metric scale, Layer 1 = World/Ground (`StaticBody3D`), Layer 2 = Player (`CharacterBody3D`).
- Project Resources**: `res://scenes/main_3d.tscn`, `res://scripts/game_state.gd`, `res://scripts/player_controller.gd`
- Collision Configuration**: Verified physics bindings (layer=1/mask=1).
- Exported Inspector Properties**: `@export var jump_velocity: float = 4.8`. Always call `move_and_slide()` inside `_physics_process(delta)`.`
