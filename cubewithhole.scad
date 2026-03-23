cube_size = 50;
hole_diameter = 20;
hole_height = 60;

color("red")
difference() {
    cube([cube_size, cube_size, cube_size], center = true);
    
    cylinder(h = hole_height, d = hole_diameter, center = true);
}