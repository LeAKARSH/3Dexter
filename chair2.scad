seat_width = 400;
seat_depth = 300;
seat_thickness = 20;
back_height = 150;
back_thickness = 40;
leg_thickness = 30;
leg_length = 100;

union() {
    // Seat
    translate([0, 0, leg_length + back_height + seat_thickness/2])
        cube([seat_width, seat_depth, seat_thickness], center = true);

    // Backrest
    translate([0, 0, leg_length + back_height/2])
        cube([seat_width, back_thickness, back_height], center = true);

    // Legs
    translate([seat_width/2 - leg_length/2, seat_depth/2 - leg_length/2, leg_thickness/2])
        cube([leg_length, leg_length, leg_thickness], center = true);
    translate([-seat_width/2 + leg_length/2, seat_depth/2 - leg_length/2, leg_thickness/2])
        cube([leg_length, leg_length, leg_thickness], center = true);
    translate([seat_width/2 - leg_length/2, -seat_depth/2 + leg_length/2, leg_thickness/2])
        cube([leg_length, leg_length, leg_thickness], center = true);
    translate([-seat_width/2 + leg_length/2, -seat_depth/2 + leg_length/2, leg_thickness/2])
        cube([leg_length, leg_length, leg_thickness], center = true);
}