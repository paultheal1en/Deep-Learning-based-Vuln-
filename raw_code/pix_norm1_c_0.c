static int pix_norm1_c(uint8_t * pix, int line_size)

{

    int s, i, j;

    uint32_t *sq = ff_squareTbl + 256;



    s = 0;

    for (i = 0; i < 16; i++) {

        for (j = 0; j < 16; j += 8) {

#if 0

            s += sq[pix[0]];

            s += sq[pix[1]];

            s += sq[pix[2]];

            s += sq[pix[3]];

            s += sq[pix[4]];

            s += sq[pix[5]];

            s += sq[pix[6]];

            s += sq[pix[7]];

#else

#if LONG_MAX > 2147483647

            register uint64_t x=*(uint64_t*)pix;

            s += sq[x&0xff];

            s += sq[(x>>8)&0xff];

            s += sq[(x>>16)&0xff];

            s += sq[(x>>24)&0xff];

            s += sq[(x>>32)&0xff];

            s += sq[(x>>40)&0xff];

            s += sq[(x>>48)&0xff];

            s += sq[(x>>56)&0xff];

#else

            register uint32_t x=*(uint32_t*)pix;

            s += sq[x&0xff];

            s += sq[(x>>8)&0xff];

            s += sq[(x>>16)&0xff];

            s += sq[(x>>24)&0xff];

            x=*(uint32_t*)(pix+4);

            s += sq[x&0xff];

            s += sq[(x>>8)&0xff];

            s += sq[(x>>16)&0xff];

            s += sq[(x>>24)&0xff];

#endif

#endif

            pix += 8;

        }

        pix += line_size - 16;

    }

    return s;

}
