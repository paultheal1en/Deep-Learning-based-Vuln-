unsigned long av_adler32_update(unsigned long adler, const uint8_t * buf,

                                unsigned int len)

{

    unsigned long s1 = adler & 0xffff;

    unsigned long s2 = adler >> 16;



    while (len > 0) {

#if CONFIG_SMALL

        while (len > 4  && s2 < (1U << 31)) {

            DO4(buf);

            len -= 4;

        }

#else

        while (len > 16 && s2 < (1U << 31)) {

            DO16(buf);

            len -= 16;

        }

#endif

        DO1(buf); len--;

        s1 %= BASE;

        s2 %= BASE;

    }

    return (s2 << 16) | s1;

}
