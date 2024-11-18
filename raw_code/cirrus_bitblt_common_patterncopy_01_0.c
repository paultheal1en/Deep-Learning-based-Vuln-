static int cirrus_bitblt_common_patterncopy(CirrusVGAState *s, bool videosrc)

{

    uint32_t patternsize;

    uint8_t *dst;

    uint8_t *src;



    dst = s->vga.vram_ptr + s->cirrus_blt_dstaddr;



    if (videosrc) {

        switch (s->vga.get_bpp(&s->vga)) {

        case 8:

            patternsize = 64;

            break;

        case 15:

        case 16:

            patternsize = 128;

            break;

        case 24:

        case 32:

        default:

            patternsize = 256;

            break;

        }

        s->cirrus_blt_srcaddr &= ~(patternsize - 1);

        if (s->cirrus_blt_srcaddr + patternsize > s->vga.vram_size) {

            return 0;

        }

        src = s->vga.vram_ptr + s->cirrus_blt_srcaddr;

    } else {

        src = s->cirrus_bltbuf;

    }



    if (blit_is_unsafe(s, true)) {

        return 0;

    }



    (*s->cirrus_rop) (s, dst, src,

                      s->cirrus_blt_dstpitch, 0,

                      s->cirrus_blt_width, s->cirrus_blt_height);

    cirrus_invalidate_region(s, s->cirrus_blt_dstaddr,

                             s->cirrus_blt_dstpitch, s->cirrus_blt_width,

                             s->cirrus_blt_height);

    return 1;

}
