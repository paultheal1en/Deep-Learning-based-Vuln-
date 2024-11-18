static void body(uint32_t ABCD[4], uint32_t X[16])

{

    int i av_unused;

    uint32_t t;

    uint32_t a = ABCD[3];

    uint32_t b = ABCD[2];

    uint32_t c = ABCD[1];

    uint32_t d = ABCD[0];



#if HAVE_BIGENDIAN

    for (i = 0; i < 16; i++)

        X[i] = av_bswap32(X[i]);

#endif



#if CONFIG_SMALL

    for (i = 0; i < 64; i++) {

        CORE(i, a, b, c, d);

        t = d;

        d = c;

        c = b;

        b = a;

        a = t;

    }

#else

#define CORE2(i)                                                        \

    CORE( i,   a,b,c,d); CORE((i+1),d,a,b,c);                           \

    CORE((i+2),c,d,a,b); CORE((i+3),b,c,d,a)

#define CORE4(i) CORE2(i); CORE2((i+4)); CORE2((i+8)); CORE2((i+12))

    CORE4(0); CORE4(16); CORE4(32); CORE4(48);

#endif



    ABCD[0] += d;

    ABCD[1] += c;

    ABCD[2] += b;

    ABCD[3] += a;

}
