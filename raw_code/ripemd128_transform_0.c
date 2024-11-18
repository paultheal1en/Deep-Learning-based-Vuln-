static void ripemd128_transform(uint32_t *state, const uint8_t buffer[64], int ext)

{

    uint32_t a, b, c, d, e, f, g, h;

    uint32_t block[16];

    int n;



    if (ext) {

        a = state[0]; b = state[1]; c = state[2]; d = state[3];

        e = state[4]; f = state[5]; g = state[6]; h = state[7];

    } else {

        a = e = state[0];

        b = f = state[1];

        c = g = state[2];

        d = h = state[3];

    }



    for (n = 0; n < 16; n++)

        block[n] = AV_RL32(buffer + 4 * n);



    for (n = 0; n < 16;) {

        ROUND128_0_TO_15(a,b,c,d,e,f,g,h);

        ROUND128_0_TO_15(d,a,b,c,h,e,f,g);

        ROUND128_0_TO_15(c,d,a,b,g,h,e,f);

        ROUND128_0_TO_15(b,c,d,a,f,g,h,e);

    }

    SWAP(a,e)



    for (; n < 32;) {

        ROUND128_16_TO_31(a,b,c,d,e,f,g,h);

        ROUND128_16_TO_31(d,a,b,c,h,e,f,g);

        ROUND128_16_TO_31(c,d,a,b,g,h,e,f);

        ROUND128_16_TO_31(b,c,d,a,f,g,h,e);

    }

    SWAP(b,f)



    for (; n < 48;) {

        ROUND128_32_TO_47(a,b,c,d,e,f,g,h);

        ROUND128_32_TO_47(d,a,b,c,h,e,f,g);

        ROUND128_32_TO_47(c,d,a,b,g,h,e,f);

        ROUND128_32_TO_47(b,c,d,a,f,g,h,e);

    }

    SWAP(c,g)



    for (; n < 64;) {

        ROUND128_48_TO_63(a,b,c,d,e,f,g,h);

        ROUND128_48_TO_63(d,a,b,c,h,e,f,g);

        ROUND128_48_TO_63(c,d,a,b,g,h,e,f);

        ROUND128_48_TO_63(b,c,d,a,f,g,h,e);

    }

    SWAP(d,h)



    if (ext) {

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;

        state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    } else {

        h += c + state[1];

        state[1] = state[2] + d + e;

        state[2] = state[3] + a + f;

        state[3] = state[0] + b + g;

        state[0] = h;

    }

}
