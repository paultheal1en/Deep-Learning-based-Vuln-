static void ff_h264_idct_dc_add_mmx2(uint8_t *dst, int16_t *block, int stride)

{

    int dc = (block[0] + 32) >> 6;

    __asm__ volatile(

        "movd          %0, %%mm0 \n\t"

        "pshufw $0, %%mm0, %%mm0 \n\t"

        "pxor       %%mm1, %%mm1 \n\t"

        "psubw      %%mm0, %%mm1 \n\t"

        "packuswb   %%mm0, %%mm0 \n\t"

        "packuswb   %%mm1, %%mm1 \n\t"

        ::"r"(dc)

    );

    __asm__ volatile(

        "movd          %0, %%mm2 \n\t"

        "movd          %1, %%mm3 \n\t"

        "movd          %2, %%mm4 \n\t"

        "movd          %3, %%mm5 \n\t"

        "paddusb    %%mm0, %%mm2 \n\t"

        "paddusb    %%mm0, %%mm3 \n\t"

        "paddusb    %%mm0, %%mm4 \n\t"

        "paddusb    %%mm0, %%mm5 \n\t"

        "psubusb    %%mm1, %%mm2 \n\t"

        "psubusb    %%mm1, %%mm3 \n\t"

        "psubusb    %%mm1, %%mm4 \n\t"

        "psubusb    %%mm1, %%mm5 \n\t"

        "movd       %%mm2, %0    \n\t"

        "movd       %%mm3, %1    \n\t"

        "movd       %%mm4, %2    \n\t"

        "movd       %%mm5, %3    \n\t"

        :"+m"(*(uint32_t*)(dst+0*stride)),

         "+m"(*(uint32_t*)(dst+1*stride)),

         "+m"(*(uint32_t*)(dst+2*stride)),

         "+m"(*(uint32_t*)(dst+3*stride))

    );

}
