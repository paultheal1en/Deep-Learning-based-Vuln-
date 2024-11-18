static void copy_bits(PutBitContext *pb, UINT8 *src, int length)

{

#if 1

    int bytes= length>>4;

    int bits= length&15;

    int i;



    for(i=0; i<bytes; i++) put_bits(pb, 16, be2me_16(((uint16_t*)src)[i]));

    put_bits(pb, bits, be2me_16(((uint16_t*)src)[i])>>(16-bits));

#else

    int bytes= length>>3;

    int bits= length&7;

    int i;



    for(i=0; i<bytes; i++) put_bits(pb, 8, src[i]);

    put_bits(pb, bits, src[i]>>(8-bits));

#endif

}
