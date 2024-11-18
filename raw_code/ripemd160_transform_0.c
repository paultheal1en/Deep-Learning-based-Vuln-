static void ripemd160_transform(uint32_t *state, const uint8_t buffer[64], int ext)

{

    uint32_t a, b, c, d, e, f, g, h, i, j;

    uint32_t block[16];

    int n;



    if (ext) {

        a = state[0]; b = state[1]; c = state[2]; d = state[3]; e = state[4];

        f = state[5]; g = state[6]; h = state[7]; i = state[8]; j = state[9];

    } else {

        a = f = state[0];

        b = g = state[1];

        c = h = state[2];

        d = i = state[3];

        e = j = state[4];

    }



    for (n = 0; n < 16; n++)

        block[n] = AV_RL32(buffer + 4 * n);



    for (n = 0; n < 16 - 1;) {

        ROUND160_0_TO_15(a,b,c,d,e,f,g,h,i,j);

        ROUND160_0_TO_15(e,a,b,c,d,j,f,g,h,i);

        ROUND160_0_TO_15(d,e,a,b,c,i,j,f,g,h);

        ROUND160_0_TO_15(c,d,e,a,b,h,i,j,f,g);

        ROUND160_0_TO_15(b,c,d,e,a,g,h,i,j,f);

    }

    ROUND160_0_TO_15(a,b,c,d,e,f,g,h,i,j);

    SWAP(a,f)



    for (; n < 32 - 1;) {

        ROUND160_16_TO_31(e,a,b,c,d,j,f,g,h,i);

        ROUND160_16_TO_31(d,e,a,b,c,i,j,f,g,h);

        ROUND160_16_TO_31(c,d,e,a,b,h,i,j,f,g);

        ROUND160_16_TO_31(b,c,d,e,a,g,h,i,j,f);

        ROUND160_16_TO_31(a,b,c,d,e,f,g,h,i,j);

    }

    ROUND160_16_TO_31(e,a,b,c,d,j,f,g,h,i);

    SWAP(b,g)



    for (; n < 48 - 1;) {

        ROUND160_32_TO_47(d,e,a,b,c,i,j,f,g,h);

        ROUND160_32_TO_47(c,d,e,a,b,h,i,j,f,g);

        ROUND160_32_TO_47(b,c,d,e,a,g,h,i,j,f);

        ROUND160_32_TO_47(a,b,c,d,e,f,g,h,i,j);

        ROUND160_32_TO_47(e,a,b,c,d,j,f,g,h,i);

    }

    ROUND160_32_TO_47(d,e,a,b,c,i,j,f,g,h);

    SWAP(c,h)



    for (; n < 64 - 1;) {

        ROUND160_48_TO_63(c,d,e,a,b,h,i,j,f,g);

        ROUND160_48_TO_63(b,c,d,e,a,g,h,i,j,f);

        ROUND160_48_TO_63(a,b,c,d,e,f,g,h,i,j);

        ROUND160_48_TO_63(e,a,b,c,d,j,f,g,h,i);

        ROUND160_48_TO_63(d,e,a,b,c,i,j,f,g,h);

    }

    ROUND160_48_TO_63(c,d,e,a,b,h,i,j,f,g);

    SWAP(d,i)



    for (; n < 75;) {

        ROUND160_64_TO_79(b,c,d,e,a,g,h,i,j,f);

        ROUND160_64_TO_79(a,b,c,d,e,f,g,h,i,j);

        ROUND160_64_TO_79(e,a,b,c,d,j,f,g,h,i);

        ROUND160_64_TO_79(d,e,a,b,c,i,j,f,g,h);

        ROUND160_64_TO_79(c,d,e,a,b,h,i,j,f,g);

    }

    ROUND160_64_TO_79(b,c,d,e,a,g,h,i,j,f);

    SWAP(e,j)



    if (ext) {

        state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;

        state[5] += f; state[6] += g; state[7] += h; state[8] += i; state[9] += j;

    } else {

        i += c + state[1];

        state[1] = state[2] + d + j;

        state[2] = state[3] + e + f;

        state[3] = state[4] + a + g;

        state[4] = state[0] + b + h;

        state[0] = i;

    }

}
