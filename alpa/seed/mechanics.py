"""Phase 2 seed content: intro Newtonian mechanics.

The concept graph below is HAND-AUTHORED — that is deliberate: Phase 1 §1.1 makes
the graph curated data, not LLM output. When LLM extraction lands (Phase 5+), it
writes proposals into the same `concept_relationships` table with
`provenance='llm:<model>', human_reviewed=False` and never touches this file.

Items are auto-gradable MCQs with difficulty *priors* b (logit scale). The
production difficulty is empirical (IRT b refit offline from attempts); these
values only cold-start item selection.
"""
from __future__ import annotations

COURSE = dict(code="MECH101", title="Intro Newtonian Mechanics",
              description="Vectors, kinematics, Newton's laws, forces, and dynamics.")

# (code, title, description, [prerequisite codes]) — listed in topological order.
CONCEPTS: list[tuple[str, str, str, list[str]]] = [
    ("vec_units", "Scalars, vectors & units",
     "Vector vs scalar quantities, components, SI units.", []),
    ("pos_disp", "Position & displacement",
     "Position coordinates and displacement as change in position.", ["vec_units"]),
    ("velocity", "Velocity",
     "Average and instantaneous velocity; displacement over time.", ["pos_disp"]),
    ("acceleration", "Acceleration",
     "Acceleration as rate of change of velocity; sign conventions.", ["velocity"]),
    ("kinematics", "Constant-acceleration equations",
     "The kinematic equations for uniformly accelerated motion.", ["acceleration"]),
    ("motion_graphs", "Position-time & velocity-time graphs",
     "Reading slopes and areas of motion graphs.", ["velocity", "acceleration"]),
    ("force_net", "Force & net force",
     "Force as interaction; net force as vector sum.", ["vec_units"]),
    ("n1", "Newton's first law",
     "Inertia: motion is unchanged when net force is zero.", ["force_net"]),
    ("n2", "Newton's second law (F = ma)",
     "Net force equals mass times acceleration.", ["acceleration", "force_net"]),
    ("n3", "Newton's third law",
     "Action-reaction pairs act on different objects.", ["n2"]),
    ("weight", "Weight (W = mg)",
     "Gravitational force near Earth's surface.", ["n2"]),
    ("contact_forces", "Normal, tension & friction",
     "Contact forces: normal force, rope tension, friction.", ["force_net"]),
    ("fbd", "Free-body diagrams",
     "Drawing all forces on the chosen object.", ["force_net", "weight", "contact_forces"]),
    ("dynamics", "Applying F = ma to problems",
     "Combining free-body diagrams with Newton's second law.", ["n2", "fbd"]),
]

# Extra typed edges beyond prerequisites (edge_type, src_code, dst_code).
EXTRA_EDGES = [
    ("SPECIAL_CASE_OF", "weight", "force_net"),
    ("RELATED", "motion_graphs", "kinematics"),
]

# Items: (concept_code, kind, stem, [choices], correct_index, b, a)
# kind: diagnostic | practice | retrieval_probe | transfer
ITEMS: list[tuple[str, str, str, list[str], int, float, float]] = [
    # ---- vec_units
    ("vec_units", "diagnostic", "Which of the following is a vector quantity?",
     ["Distance", "Speed", "Displacement", "Mass"], 2, -1.2, 1.0),
    ("vec_units", "practice", "What is the SI unit of force?",
     ["kg", "N", "J", "Pa"], 1, -0.8, 1.0),
    ("vec_units", "practice", "A vector of magnitude 5 points along +x. Its y-component is:",
     ["0", "5", "-5", "3"], 0, -0.4, 1.0),
    ("vec_units", "practice", "3 m east plus 4 m north gives a resultant of magnitude:",
     ["7 m", "5 m", "1 m", "12 m"], 1, 0.2, 1.2),
    ("vec_units", "practice", "A = (3, 0) and B = (0, 4). What is |A + B| - |A - B|?",
     ["0", "5", "7", "1"], 0, 0.9, 1.2),
    # ---- pos_disp
    ("pos_disp", "diagnostic", "Displacement is best described as:",
     ["The total path length travelled", "The change in position", "Speed multiplied by time, always",
      "Distance squared"], 1, -1.0, 1.0),
    ("pos_disp", "practice", "You walk 10 m east then 4 m west. Your displacement from the start is:",
     ["14 m west", "6 m east", "6 m west", "14 m east"], 1, -0.5, 1.0),
    ("pos_disp", "practice", "After a round trip returning to the start, the displacement is:",
     ["Zero", "Equal to the distance", "Impossible to tell", "The negative of the distance"], 0, -0.8, 1.0),
    ("pos_disp", "practice", "x = 2 m at t = 0 and x = -3 m at t = 5 s. The displacement is:",
     ["-5 m", "5 m", "-1 m", "1 m"], 0, 0.0, 1.0),
    ("pos_disp", "practice", "A particle goes from x = -4 m to x = 3 m, then back to x = -1 m. Total displacement?",
     ["-5 m", "3 m", "11 m", "-3 m"], 1, 0.7, 1.1),
    # ---- velocity
    ("velocity", "diagnostic", "Average velocity is defined as:",
     ["Distance over time, always", "Displacement over time", "Speed times time",
      "Acceleration times time"], 1, -1.0, 1.0),
    ("velocity", "practice", "A car travels 100 m east in 5 s. Its average velocity is:",
     ["20 m/s east", "500 m/s", "0.05 m/s", "20 m/s, no direction"], 0, -0.6, 1.0),
    ("velocity", "practice", "Instantaneous velocity is:",
     ["The average over the whole trip", "The velocity at a single instant",
      "Always equal to speed", "Always positive"], 1, -0.2, 1.0),
    ("velocity", "practice", "2 m/s for 3 s then 4 m/s for 2 s, same direction. Average speed?",
     ["3.0 m/s", "2.8 m/s", "3.2 m/s", "6.0 m/s"], 1, 0.5, 1.2),
    ("velocity", "practice", "Half a lap of a 200 m circular track in 25 s. Magnitude of average velocity?",
     ["8 m/s", "4 m/s", "1.6 m/s", "0 m/s"], 2, 0.9, 1.2),
    # ---- acceleration
    ("acceleration", "diagnostic", "Acceleration is the rate of change of:",
     ["Position", "Velocity", "Distance", "Force"], 1, -1.0, 1.0),
    ("acceleration", "practice", "A car speeds up from 0 to 20 m/s in 4 s. Its acceleration is:",
     ["5 m/s^2", "80 m/s^2", "0.2 m/s^2", "16 m/s^2"], 0, -0.5, 1.0),
    ("acceleration", "practice", "An object moves in the +x direction while slowing down. Its acceleration is:",
     ["Positive", "Negative", "Zero", "Impossible to tell"], 1, 0.3, 1.2),
    ("acceleration", "retrieval_probe", "At the top of its flight, a ball thrown upward has acceleration:",
     ["Zero", "g, downward", "g, upward", "Depends on the throw"], 1, 0.6, 1.0),
    ("acceleration", "practice", "A car's velocity changes from +15 m/s to -5 m/s in 4 s. Average acceleration?",
     ["-5 m/s^2", "5 m/s^2", "-2.5 m/s^2", "20 m/s^2"], 0, 0.9, 1.2),
    # ---- kinematics
    ("kinematics", "diagnostic", "The equation v = v0 + at applies when:",
     ["Acceleration is constant", "Always", "v0 = 0", "No forces act"], 0, -0.4, 1.0),
    ("kinematics", "practice", "Starting from rest with a = 2 m/s^2, the speed after 3 s is:",
     ["6 m/s", "9 m/s", "2 m/s", "18 m/s"], 0, -0.1, 1.0),
    ("kinematics", "practice", "From rest with a = 2 m/s^2, the displacement after 3 s is:",
     ["9 m", "6 m", "18 m", "3 m"], 0, 0.3, 1.0),
    ("kinematics", "practice", "With v0 = 10 m/s and a = -2 m/s^2, time to come to rest is:",
     ["5 s", "2 s", "20 s", "10 s"], 0, 0.5, 1.0),
    ("kinematics", "practice", "A car at 10 m/s brakes with a = -5 m/s^2. Stopping distance?",
     ["10 m", "5 m", "20 m", "2 m"], 0, 0.9, 1.2),
    # ---- motion_graphs
    ("motion_graphs", "diagnostic", "The slope of a position-time graph gives:",
     ["Acceleration", "Velocity", "Distance", "Force"], 1, -0.6, 1.0),
    ("motion_graphs", "practice", "The slope of a velocity-time graph gives:",
     ["Velocity", "Acceleration", "Displacement", "Jerk only"], 1, -0.5, 1.0),
    ("motion_graphs", "practice", "The area under a velocity-time graph gives:",
     ["Acceleration", "Displacement", "Speed", "Force"], 1, -0.3, 1.0),
    ("motion_graphs", "practice", "A horizontal line on a velocity-time graph means:",
     ["The object is at rest", "Constant velocity", "Constant acceleration", "Increasing velocity"], 1, -0.1, 1.0),
    # ---- force_net
    ("force_net", "diagnostic", "A force is best described as:",
     ["A push or pull (an interaction)", "Stored energy", "Motion", "Mass times velocity"], 0, -1.0, 1.0),
    ("force_net", "practice", "The net force on an object is:",
     ["The largest force acting", "The vector sum of all forces", "The product of the forces",
      "Always zero"], 1, -0.7, 1.0),
    ("force_net", "practice", "Forces of 6 N right and 4 N left act on one object. Net force?",
     ["10 N right", "2 N right", "2 N left", "24 N"], 1, -0.2, 1.0),
    ("force_net", "practice", "If the net force on an object is zero, the object:",
     ["Must be at rest", "Must accelerate", "Has constant velocity (possibly zero)", "Must move right"], 2, 0.3, 1.1),
    # ---- n1
    ("n1", "diagnostic", "Newton's first law states that motion is unchanged when:",
     ["The net force is zero", "Only gravity acts", "Friction is present", "Mass is large"], 0, -0.8, 1.0),
    ("n1", "practice", "A book rests on a table. It does not move because:",
     ["No forces act on it", "The forces on it balance", "Gravity is absent", "The table pushes harder than gravity"], 1, -0.3, 1.0),
    ("n1", "practice", "A puck sliding on frictionless ice will:",
     ["Naturally slow down", "Continue at constant velocity", "Speed up", "Stop after a fixed time"], 1, 0.1, 1.0),
    # ---- n2
    ("n2", "diagnostic", "Newton's second law says net force equals mass times:",
     ["Velocity", "Acceleration", "Displacement", "Speed"], 1, -0.8, 1.0),
    ("n2", "practice", "A net force of 12 N acts on a 3 kg mass. Acceleration?",
     ["4 m/s^2", "36 m/s^2", "0.25 m/s^2", "9 m/s^2"], 0, -0.4, 1.0),
    ("n2", "practice", "The same net force acts on two objects. The more massive one has:",
     ["Larger acceleration", "Smaller acceleration", "The same acceleration", "Zero acceleration"], 1, -0.1, 1.0),
    ("n2", "practice", "m = 2 kg and a = 3 m/s^2. The net force is:",
     ["6 N", "1.5 N", "5 N", "9 N"], 0, -0.2, 1.0),
    ("n2", "retrieval_probe", "An object moves in a straight line at constant velocity. The net force on it is:",
     ["Zero", "Equal to ma > 0", "Equal to its weight", "It depends on the speed"], 0, 0.6, 1.2),
    ("n2", "practice", "Two forces act on a 4 kg mass: 20 N right and 8 N left. Acceleration?",
     ["3 m/s^2 right", "7 m/s^2 right", "2 m/s^2 right", "5 m/s^2 right"], 0, 0.8, 1.1),
    # ---- n3
    ("n3", "diagnostic", "An action-reaction pair always acts on:",
     ["The same object", "Two different objects", "The heavier object", "The object with the larger force"], 1, -0.3, 1.0),
    ("n3", "practice", "Earth pulls a falling apple downward. The reaction force is:",
     ["The apple's weight", "The apple pulling Earth upward", "The normal force", "Air resistance"], 1, 0.2, 1.2),
    ("n3", "practice", "Action-reaction forces do not cancel because they:",
     ["Are unequal", "Act on different objects", "Act at different times", "Only exist in contact"], 1, 0.1, 1.0),
    ("n3", "transfer", "A small car collides with a large truck. During the collision:",
     ["The car feels the larger force", "The truck feels the larger force",
      "Both feel equal-magnitude forces", "The faster vehicle feels the larger force"], 2, 0.5, 1.2),
    # ---- weight
    ("weight", "diagnostic", "Weight is:",
     ["Mass", "The gravitational force mg", "Volume", "Mass times velocity"], 1, -0.9, 1.0),
    ("weight", "practice", "The weight of a 5 kg object (g = 10 m/s^2) is:",
     ["5 N", "50 N", "0.5 N", "500 N"], 1, -0.5, 1.0),
    ("weight", "practice", "On the Moon, an object's mass compared to Earth is:",
     ["Smaller", "The same", "Larger", "Zero"], 1, 0.0, 1.0),
    # ---- contact_forces
    ("contact_forces", "diagnostic", "The normal force acts:",
     ["Parallel to the surface", "Perpendicular to the surface", "Always straight up",
      "Always opposite gravity"], 1, -0.5, 1.0),
    ("contact_forces", "practice", "Tension is the force transmitted through:",
     ["A rope or string", "A rigid surface", "Air", "A spring only"], 0, -0.7, 1.0),
    ("contact_forces", "practice", "Kinetic friction acts:",
     ["In the direction of motion", "Opposite the relative sliding", "Perpendicular to the surface",
      "Always equal to the normal force"], 1, -0.2, 1.0),
    ("contact_forces", "practice", "A book rests on a level table. The normal force magnitude is:",
     ["Greater than the weight", "Equal to the weight", "Less than the weight", "Zero"], 1, 0.2, 1.1),
    # ---- fbd
    ("fbd", "diagnostic", "A free-body diagram shows the forces acting on:",
     ["All objects in the problem", "The chosen object only", "Only gravity", "Only contact forces"], 1, -0.6, 1.0),
    ("fbd", "practice", "The FBD of a book resting on a level table contains:",
     ["Weight down and normal force up", "Weight only", "Weight, normal force and a push",
      "Weight, normal force and ma"], 0, -0.2, 1.0),
    ("fbd", "retrieval_probe", "A ball is at the top of its vertical flight (no air resistance). Its FBD contains:",
     ["Gravity, downward, only", "Gravity plus an upward 'throw force'", "No forces",
      "Gravity plus ma"], 0, 0.5, 1.3),
    ("fbd", "practice", "A block is pulled right at constant speed against friction. Horizontal forces are:",
     ["The pull only", "Pull greater than friction", "Pull equal to friction", "Friction only"], 2, 0.4, 1.0),
    # ---- dynamics
    ("dynamics", "diagnostic", "A 10 N horizontal force pulls a 2 kg block on a frictionless floor. Acceleration?",
     ["5 m/s^2", "20 m/s^2", "0.2 m/s^2", "12 m/s^2"], 0, 0.2, 1.0),
    ("dynamics", "practice", "Same block, now with 4 N of friction opposing a 10 N pull. Acceleration?",
     ["3 m/s^2", "5 m/s^2", "7 m/s^2", "2 m/s^2"], 0, 0.6, 1.0),
    ("dynamics", "transfer", "An elevator accelerates upward. The apparent weight of a passenger is:",
     ["Less than mg", "Greater than mg", "Equal to mg", "Zero"], 1, 0.9, 1.2),
    ("dynamics", "practice", "A 5 kg mass hangs at rest from a rope (g = 10 m/s^2). The tension is:",
     ["0 N", "50 N", "25 N", "5 N"], 1, 0.3, 1.0),
]

# Challenge items anchored at/above mastery level (b ~ 1.2-1.4). Mastery claims are
# only measurable if items exist where mastery-level students sit near p ~ 0.5-0.7;
# without these, information collapses before mastery is reachable (test-design
# requirement, not content flavor).
CHALLENGE_ITEMS: list[tuple[str, str, str, list[str], int, float, float]] = [
    ("vec_units", "practice", "A = (3, 4). The unit vector along A is:",
     ["(0.6, 0.8)", "(3, 4)", "(0.8, 0.6)", "(1, 1)"], 0, 1.3, 1.2),
    ("pos_disp", "practice", "x(t) = 2t^2 - 3t (meters). Displacement from t = 0 to t = 2 s?",
     ["4 m", "2 m", "8 m", "-3 m"], 1, 1.2, 1.2),
    ("velocity", "practice", "x(t) = t^3 (meters). Instantaneous velocity at t = 2 s?",
     ["8 m/s", "6 m/s", "12 m/s", "4 m/s"], 2, 1.3, 1.2),
    ("acceleration", "practice", "v(t) = 4t^2 - 2t (m/s). Acceleration at t = 1 s?",
     ["6 m/s^2", "2 m/s^2", "4 m/s^2", "8 m/s^2"], 0, 1.3, 1.2),
    ("kinematics", "practice", "A stone falls from rest (g = 10 m/s^2). Distance covered during the 3rd second?",
     ["45 m", "25 m", "20 m", "30 m"], 1, 1.3, 1.2),
    ("motion_graphs", "practice", "The area under an acceleration-time graph gives:",
     ["Displacement", "Force", "The change in velocity", "Jerk"], 2, 1.2, 1.2),
    ("force_net", "practice", "Forces 5 N east, 3 N west and 4 N north act on one object. Net force magnitude?",
     ["12 N", "4.5 N", "2 N", "6 N"], 1, 1.3, 1.2),
    ("n1", "practice", "A bus brakes hard and a standing passenger lurches forward. Best explanation:",
     ["A forward force pushes the passenger", "The passenger's inertia maintains their motion",
      "The bus pushes the passenger forward", "Gravity tilts the passenger"], 1, 1.2, 1.2),
    ("n2", "practice", "A force F gives mass m an acceleration of 6 m/s^2. The same F on mass 3m gives:",
     ["6 m/s^2", "18 m/s^2", "2 m/s^2", "3 m/s^2"], 2, 1.3, 1.2),
    ("n3", "practice", "A horse pulls a cart and the cart accelerates. This is consistent with Newton's third law because:",
     ["The pull on the cart exceeds the friction resisting the cart",
      "Action-reaction forces cancel only at constant speed",
      "The horse exerts a larger force than the cart",
      "Gravity adds an extra force on the cart"], 0, 1.4, 1.2),
    ("weight", "practice", "On Mars (g = 4 m/s^2) a 10 kg object has:",
     ["Mass 10 kg, weight 40 N", "Mass 4 kg, weight 40 N",
      "Mass 10 kg, weight 100 N", "Mass 4 kg, weight 10 N"], 0, 1.2, 1.2),
    ("contact_forces", "practice", "A 2 kg block rests on a 30 degree incline (g = 10 m/s^2). The normal force is:",
     ["17.3 N", "20 N", "10 N", "11.5 N"], 0, 1.4, 1.2),
    ("fbd", "practice", "A block slides down a rough incline at constant speed. Which relation holds?",
     ["Friction = mg sin(theta)", "Friction = mg cos(theta)",
      "Friction > mg", "Normal force = mg"], 0, 1.4, 1.2),
    ("dynamics", "practice", "A 3 kg block is pulled by 15 N against 6 N of friction, from rest. Speed after 2 s?",
     ["6 m/s", "3 m/s", "10 m/s", "2 m/s"], 0, 1.4, 1.2),
]

ITEMS = ITEMS + CHALLENGE_ITEMS


GLOBAL_MU = -0.5     # population prior mean ability for a fresh concept
GLOBAL_SIGMA = 1.0   # population prior sd
