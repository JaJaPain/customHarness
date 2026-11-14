extends CharacterBody3D
class_name NebulaPaddle3D

signal ball_deflected(hit_offset: float)
signal powerup_collected(powerup_type: String)

@export var move_speed: float = 16.0
@export var arena_half_width: float = 8.4
@export var base_width: float = 3.0
@export var use_mouse_control: bool = false

var width_multiplier: float = 1.0
var wide_timer: float = 0.0

func _ready() -> void:
	add_to_group("paddle")

func _physics_process(delta: float) -> void:
	if wide_timer > 0.0:
		wide_timer -= delta
		if wide_timer <= 0.0:
			set_wide_paddle(false)

	var dir: float = 0.0
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT) or Input.is_action_pressed("ui_left"):
		dir -= 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT) or Input.is_action_pressed("ui_right"):
		dir += 1.0

	velocity.x = dir * move_speed
	velocity.y = 0.0
	velocity.z = 0.0
	move_and_slide()
	var limit: float = maxf(2.0, arena_half_width - (base_width * width_multiplier * 0.5))
	global_position.x = clampf(global_position.x, -limit, limit)
	global_position.y = 0.3
	global_position.z = 8.5

func get_deflection_direction(hit_world_x: float) -> Vector3:
	var half_w: float = maxf(0.5, (base_width * width_multiplier) * 0.5)
	var rel: float = clampf((hit_world_x - global_position.x) / half_w, -1.0, 1.0)
	var max_angle: float = deg_to_rad(62.0)
	var angle: float = rel * max_angle
	var dir := Vector3(sin(angle), 0.0, -cos(angle)).normalized()
	ball_deflected.emit(rel)
	return dir

func set_wide_paddle(enabled: bool, duration: float = 10.0) -> void:
	width_multiplier = 1.55 if enabled else 1.0
	wide_timer = duration if enabled else 0.0
	scale.x = width_multiplier
