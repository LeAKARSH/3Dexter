body_radius = 25;
body_height = 60;
lid_radius = 25;
lid_height = 10;
wall_thickness = 2;
handle_radius = 5;
handle_length = 15;

union() {
    difference() {
        cylinder(r = body_radius, h = body_height, $fn = 10);
        translate([0, 0, wall_thickness])
            cylinder(r = body_radius - wall_thickness, h = body_height, $fn = 10);
    }

    translate([0, 0, body_height])
    difference() {
        cylinder(r = lid_radius, h = lid_height, $fn = 10);
        translate([0, 0, wall_thickness])
            cylinder(r = lid_radius - wall_thickness, h = lid_height, $fn = 10);
    }

    translate([body_radius - 2, 0, body_height / 2])
    rotate([0, 90, 0])
    union() {
        cylinder(r = handle_radius, h = handle_length, $fn = 50);
        translate([0, 0, handle_length])
            sphere(r = handle_radius, $fn = 50);
    }
}