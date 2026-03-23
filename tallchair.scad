seat_width = 40;
seat_depth = 20;
seat_height = 60;
back_thickness = 5;
leg_height = 40;
leg_radius = 3;

union() {
    // Seat
    translate([0, 0, leg_height + seat_height / 2])
    cube([seat_width, seat_depth, seat_height], center = true);

    // Backrest
    translate([0, -seat_depth / 2 - back_thickness / 2, leg_height / 2])
    cube([seat_width, back_thickness, back_thickness], center = true);

    // Legs
    translate([seat_width / 4, seat_depth / 4, 0]) cylinder(r = leg_radius, h = leg_height, center = true);
    translate([-seat_width / 4, seat_depth / 4, 0]) cylinder(r = leg_radius, h = leg_height, center = true);
    translate([seat_width / 4, -seat_depth / 4, 0]) cylinder(r = leg_radius, h = leg_height, center = true);
    translate([-seat_width / 4, -seat_depth / 4, 0]) cylinder(r = leg_radius, h = leg_height, center = true);
}