mug_radius = 25;
mug_height = 40;
wall_thickness = 2;
handle_radius = 5;
handle_length = 30;

union() {
    difference() {
        cylinder(r = mug_radius, h = mug_height, $fn = 10);
        translate([0, 0, -1])
            cylinder(r = mug_radius - wall_thickness, h = mug_height + 2, $fn = 10);
    }

    translate([0, 0, mug_height])
        cylinder(r = handle_radius, h = handle_length, $fn = 50);

    translate([0, 0, mug_height])
        rotate([0, 90, 0])
            cylinder(r = handle_radius + 1, h = 5, $fn = 50);
}