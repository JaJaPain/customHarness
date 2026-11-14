extends CharacterBody3D
class_name PlayerController3D

signal health_changed(new_health: int)
signal interacted(target_node: Node3D)

@export var move_speed: float = 6.0
@export var jump_velocity: float = 4.8
@export var max_health: int = 100

@onready var camera_3d: Camera3D = $Camera3D

var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)
var current_health: int = 100

func _ready() -> void:
	current_health = max_health

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= gravity * delta

	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = jump_velocity

	var input_dir := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		velocity.x = direction.x * move_speed
		velocity.z = direction.z * move_speed
	else:
		velocity.x = move_toward(velocity.x, 0, move_speed)
		velocity.z = move_toward(velocity.z, 0, move_speed)

	move_and_slide()
