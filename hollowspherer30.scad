outer_radius = 30;
wall_thickness = 3;

difference() {
    sphere(r = outer_radius, $fn = 100);
    sphere(r = outer_radius - wall_thickness, $fn = 100);
}