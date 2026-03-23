seat_width = 40;
seat_depth = 30;
seat_height = 10;
back_thickness = 2;
seat_thickness = 2;
leg_height = 10;
leg_thickness = 2;
leg_spacing = 10;

union() {
    // Seat
    cube([seat_width, seat_depth, seat_height]);
    
    // Backrest
    translate([0, 0, seat_height])
    cube([seat_width, back_thickness, seat_height]);
    
    // Seat Backrest
    translate([0, -seat_depth, seat_height])
    cube([seat_width, seat_depth, seat_height]);
    
    // Legs
    translate([seat_width/2 - leg_thickness/2, seat_depth/2 - leg_thickness/2, -leg_height])
    cube([leg_thickness, leg_thickness, leg_height]);
    
    translate([seat_width/2 - leg_thickness/2, seat_depth/2 - leg_thickness/2, -leg_height])
    cube([leg_thickness, leg_thickness, leg_height]);
    
    translate([seat_width/2 - leg_thickness/2, seat_depth/2 - leg_thickness/2, -leg_height])
    cube([leg_thickness, leg_thickness, leg_height]);
    
    translate([seat_width/2 - leg_thickness/2, seat_depth/2 - leg_thickness/2, -leg_height])
    cube([leg_thickness, leg_thickness, leg_height]);
}