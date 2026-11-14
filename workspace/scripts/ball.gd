extends CharacterBody3D
class_name NebulaBall3D

signal brick_hit(brick_node: Node, impact_pos: Vector3)
signal paddle_bounced(impact_pos: Vector3)
signal ball_lost(ball_node: Node)

@export var base_speed: float = 13.5
@export var max_speed: float = 25.0
@export var speed_ramp_per_combo: float = 0.45
@export var arena_bounds_x: float = 9.6
@export var arena_top_z: float = -9.8
@export var arena_bottom_z: float = 11.2

var current_speed: float = 13.5
var move_dir: Vector3 = Vector3(0.25, 0.0, -1.0).normalized()
var is_launched: bool = false
var plasma_piercer_active: bool = false
var plasma_timer: float = 0.0
var combo_count: int = 0

func _ready() -> void:
	add_to_group("balls")
	current_speed = base_speed

func launch(initial_dir: Vector3 = Vector3(0.2, 0.0, -1.0)) -> void:
	move_dir = Vector3(initial_dir.x, 0.0, initial_dir.z).normalized()
	is_launched = true

func activate_plasma_piercer(duration: float = 6.0) -> void:
	plasma_piercer_active = true
	plasma_timer = duration

func _physics_process(delta: float) -> void:
	if not is_launched:
		return

	if plasma_piercer_active:
		plasma_timer -= delta
		if plasma_timer <= 0.0:
			plasma_piercer_active = false

	var step_vec: Vector3 = move_dir * current_speed * delta
	var collision: KinematicCollision3D = move_and_collide(step_vec)
	if collision:
		var collider: Object = collision.get_collider()
		var hit_pos: Vector3 = collision.get_position()
		var normal: Vector3 = collision.get_normal()
		normal.y = 0.0
		if normal.length_squared() < 0.001:
			normal = Vector3(0.0, 0.0, 1.0)
		else:
			normal = normal.normalized()

		if collider and collider.has_method("get_deflection_direction"):
			move_dir = collider.get_deflection_direction(hit_pos.x)
			combo_count = 0
			paddle_bounced.emit(hit_pos)
		elif collider and collider.has_method("apply_ball_damage"):
			combo_count += 1
			current_speed = minf(max_speed, base_speed + float(combo_count) * speed_ramp_per_combo)
			collider.apply_ball_damage(2 if plasma_piercer_active else 1, hit_pos)
			brick_hit.emit(collider, hit_pos)
			if not plasma_piercer_active:
				move_dir = move_dir.bounce(normal).normalized()
		else:
			move_dir = move_dir.bounce(normal).normalized()

	# Clamp to 3D containment rails
	if global_position.x <= -arena_bounds_x and move_dir.x < 0.0:
		move_dir.x = -move_dir.x
	elif global_position.x >= arena_bounds_x and move_dir.x > 0.0:
		move_dir.x = -move_dir.x
	if global_position.z <= arena_top_z and move_dir.z < 0.0:
		move_dir.z = -move_dir.z
	if absf(move_dir.z) < 0.18:
		move_dir.z = -0.25 if move_dir.z <= 0.0 else 0.25
		move_dir = move_dir.normalized()
	global_position.y = 0.35

	if global_position.z > arena_bottom_z:
		is_launched = false
		ball_lost.emit(self)
