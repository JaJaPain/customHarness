extends Area3D
class_name NebulaPowerUp3D

signal activated(powerup_type: String)

# Types: 'multi_ball', 'wide_paddle', 'plasma_piercer'
@export var powerup_type: String = "multi_ball"
@export var fall_speed: float = 5.5
@export var spin_speed: float = 2.8

func _ready() -> void:
	add_to_group("powerups")
	body_entered.connect(_on_body_entered)

func _physics_process(delta: float) -> void:
	global_position.z += fall_speed * delta
	rotate_y(spin_speed * delta)
	if global_position.z > 11.5:
		queue_free()

func _on_body_entered(body: Node3D) -> void:
	if body and body.is_in_group("paddle"):
		activated.emit(powerup_type)
		queue_free()
