gear_radius = 20;
gear_thickness = 5;
tooth_depth = 2;
tooth_width = 6;
num_tooth = 16;

union() {
    difference() {
        cylinder(r = gear_radius, h = gear_thickness, $fn = 10);
        
        for (i = [0 : num_tooth - 1]) {
            rotate([0, 0, i * (360 / num_tooth)])
            translate([gear_radius, 0, gear_thickness / 2])
            rotate([0, 90, 0])
            cube([tooth_width, tooth_depth, gear_thickness], center = true);
        }
    }
}