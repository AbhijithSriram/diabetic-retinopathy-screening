function x = triangular_draw(a, m, b)
%TRIANGULAR_DRAW  One draw from a triangular(a, mode m, b) distribution.
u = rand();
fc = (m - a) / (b - a);
if u < fc
    x = a + sqrt(u * (b - a) * (m - a));
else
    x = b - sqrt((1 - u) * (b - a) * (b - m));
end
end
