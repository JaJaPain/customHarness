extends StaticBody3D
class_name NebulaBrick3D

signal brick_destroyed(brick: Node3D, tier: int, score_value: int, world_pos: Vector3)
signal brick_cracked(brick: Node3D, world_pos: Vector3)

enum BrickTier {
	STANDARD_NEON = 1,
	ARMORED_REACTOR = 2,
	EXPLOSIVE_CORE = 3,
}

@export var brick_tier: int = BrickTier.STANDARD_NEON
@export var hit_points: int = 1
@export var score_value: int = 100
@export var explosion_radius: float = 3.2
@export var powerup_drop_chance: float = 0.28

var _is_destroyed: bool = false

func _ready() -> void:
	add_to_group("bricks")
	if brick_tier == BrickTier.ARMORED_REACTOR:
		hit_points = max(2, hit_points)
		score_value = 250
	elif brick_tier == BrickTier.EXPLOSIVE_CORE:
		hit_points = 1
		score_value = 350

func apply_ball_damage(amount: int, impact_pos: Vector3 = Vector3.ZERO) -> void:
	if _is_destroyed:
		return
	hit_points -= amount
	if hit_points > 0:
		_shift_cracked_appearance()
		brick_cracked.emit(self, global_position)
	else:
		_destroy_brick()

func _shift_cracked_appearance() -> void:
	scale = scale * 0.94

func _destroy_brick() -> void:
	if _is_destroyed:
		return
	_is_destroyed = true
	var pos: Vector3 = global_position
	if brick_tier == BrickTier.EXPLOSIVE_CORE:
		_detonate_adjacent_bricks(pos)
	brick_destroyed.emit(self, brick_tier, score_value, pos)
	queue_free()

func _detonate_adjacent_bricks(origin: Vector3) -> void:
	for node in get_tree().get_nodes_in_group("bricks"):
		if node != self and node is Node3D and is_instance_valid(node):
			var other := node as Node3D
			if other.global_position.distance_to(origin) <= explosion_radius:
				if other.has_method("apply_ball_damage"):
					other.apply_ball_damage(5, origin)
