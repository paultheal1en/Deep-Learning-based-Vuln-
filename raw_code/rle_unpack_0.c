static int rle_unpack(unsigned char *src, unsigned char *dest, int len)

{

    unsigned char *ps;

    unsigned char *pd;

    int i, l;



    ps = src;

    pd = dest;

    if (len & 1)

        *pd++ = *ps++;



    len >>= 1;

    i = 0;

    do {

        l = *ps++;

        if (l & 0x80) {

            l = (l & 0x7F) * 2;

            memcpy(pd, ps, l);

            ps += l;

            pd += l;

        } else {

            for (i = 0; i < l; i++) {

                *pd++ = ps[0];

                *pd++ = ps[1];

            }

            ps += 2;

        }

        i += l;

    } while (i < len);



    return (ps - src);

}
