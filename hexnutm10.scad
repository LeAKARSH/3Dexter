nut_diameter = 12.7;
nut_height = 6.4;
thread_depth = 1.6;
$fn = 100;

module m10_hex_nut() {
    difference() {
        // Hex body
        cylinder(r = nut_diameter / sqrt(3), h = nut_height, $fn = 6);
        
        // Hex hole
        cylinder(r = nut_diameter / sqrt(3) - thread_depth, h = nut_height, $fn = 6);
        
        // Hex flange
        cylinder(r = nut_diameter / 2, h = nut_height, $fn = 6);
    }
}

m10_hex_nut();