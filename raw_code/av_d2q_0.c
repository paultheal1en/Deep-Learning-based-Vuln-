AVRational av_d2q(double d, int max)

{

    AVRational a;

    int exponent;

    int64_t den;

    if (isnan(d))

        return (AVRational) { 0,0 };

    if (fabs(d) > INT_MAX + 3LL)

        return (AVRational) { d < 0 ? -1 : 1, 0 };

    frexp(d, &exponent);

    exponent = FFMAX(exponent-1, 0);

    den = 1LL << (61 - exponent);

    // (int64_t)rint() and llrint() do not work with gcc on ia64 and sparc64

    av_reduce(&a.num, &a.den, floor(d * den + 0.5), den, max);

    if ((!a.num || !a.den) && d && max>0 && max<INT_MAX)

        av_reduce(&a.num, &a.den, floor(d * den + 0.5), den, INT_MAX);



    return a;

}
