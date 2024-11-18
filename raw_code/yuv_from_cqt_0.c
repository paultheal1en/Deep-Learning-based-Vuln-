static void yuv_from_cqt(ColorFloat *c, const FFTComplex *v, float gamma, int len)

{

    int x;

    for (x = 0; x < len; x++) {

        float r, g, b;

        r = calculate_gamma(FFMIN(1.0f, v[x].re), gamma);

        g = calculate_gamma(FFMIN(1.0f, 0.5f * (v[x].re + v[x].im)), gamma);

        b = calculate_gamma(FFMIN(1.0f, v[x].im), gamma);

        c[x].yuv.y = 16.0f + 65.481f * r + 128.553f * g + 24.966f * b;

        c[x].yuv.u = 128.0f - 37.797f * r - 74.203f * g + 112.0f * b;

        c[x].yuv.v = 128.0f + 112.0f * r - 93.786f * g - 18.214 * b;

    }

}
