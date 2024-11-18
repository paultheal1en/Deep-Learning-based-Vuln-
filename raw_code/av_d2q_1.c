AVRational av_d2q(double d, int max)

{

    AVRational a;

#define LOG2  0.69314718055994530941723212145817656807550013436025

    int exponent;

    int64_t den;

    if (isnan(d))

        return (AVRational) { 0,0 };

    if (isinf(d))

        return (AVRational) { d < 0 ? -1 : 1, 0 };

    exponent = FFMAX( (int)(log(fabs(d) + 1e-20)/LOG2), 0);

    den = 1LL << (61 - exponent);

    av_reduce(&a.num, &a.den, (int64_t)(d * den + 0.5), den, max);



    return a;

}
