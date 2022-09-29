```python
	    def Geometric2Conic(self, ellipse):
        """ Calculate the elliptic equation coefficients """
        # Ax ^ 2 + Bxy + Cy ^ 2 + Dx + Ey + F
        (x0, y0), (bb, aa), phi_b_deg = ellipse

        # Semimajor and semiminor axes
        a, b = aa / 2, bb / 2
        # Convert phi_b from deg to rad
        phi_b_rad = phi_b_deg * np.pi / 180.0
        # Major axis unit vector
        ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

        # Useful intermediates
        a2 = a * a
        b2 = b * b

        # Conic parameters
        if a2 > 0 and b2 > 0:
            A = ax * ax / a2 + ay * ay / b2
            B = 2 * ax * ay / a2 - 2 * ax * ay / b2
            C = ay * ay / a2 + ax * ax / b2
            D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
            E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
            F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + \
                (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1
        else:
            # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
            A, B, C, D, E, F = (1, 0, 1, 0, 0, -1e-6)
```