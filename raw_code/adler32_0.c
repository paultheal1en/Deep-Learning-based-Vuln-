static uint32_t adler32(uint32_t adler, const uint8_t *buf, unsigned int len)

{

    unsigned long s1 = adler & 0xffff;

    unsigned long s2 = (adler >> 16) & 0xffff;

    int k;



    if (buf == NULL) return 1L;



    while (len > 0) {

        k = len < NMAX ? len : NMAX;

        len -= k;

        while (k >= 16) {

            DO16(buf);

            k -= 16;

        }

        if (k != 0) do {

            DO1(buf);

        } while (--k);

        s1 %= BASE;

        s2 %= BASE;

    }

    return (s2 << 16) | s1;

}
