static void vga_get_text_resolution(VGACommonState *s, int *pwidth, int *pheight,

                                    int *pcwidth, int *pcheight)

{

    int width, cwidth, height, cheight;



    /* total width & height */

    cheight = (s->cr[VGA_CRTC_MAX_SCAN] & 0x1f) + 1;

    cwidth = 8;

    if (!(s->sr[VGA_SEQ_CLOCK_MODE] & VGA_SR01_CHAR_CLK_8DOTS)) {

        cwidth = 9;

    }

    if (s->sr[VGA_SEQ_CLOCK_MODE] & 0x08) {

        cwidth = 16; /* NOTE: no 18 pixel wide */

    }

    width = (s->cr[VGA_CRTC_H_DISP] + 1);

    if (s->cr[VGA_CRTC_V_TOTAL] == 100) {

        /* ugly hack for CGA 160x100x16 - explain me the logic */

        height = 100;

    } else {

        height = s->cr[VGA_CRTC_V_DISP_END] |

            ((s->cr[VGA_CRTC_OVERFLOW] & 0x02) << 7) |

            ((s->cr[VGA_CRTC_OVERFLOW] & 0x40) << 3);

        height = (height + 1) / cheight;

    }



    *pwidth = width;

    *pheight = height;

    *pcwidth = cwidth;

    *pcheight = cheight;

}
