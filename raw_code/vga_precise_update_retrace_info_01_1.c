static void vga_precise_update_retrace_info(VGACommonState *s)

{

    int htotal_chars;

    int hretr_start_char;

    int hretr_skew_chars;

    int hretr_end_char;



    int vtotal_lines;

    int vretr_start_line;

    int vretr_end_line;



    int dots;

#if 0

    int div2, sldiv2;

#endif

    int clocking_mode;

    int clock_sel;

    const int clk_hz[] = {25175000, 28322000, 25175000, 25175000};

    int64_t chars_per_sec;

    struct vga_precise_retrace *r = &s->retrace_info.precise;



    htotal_chars = s->cr[VGA_CRTC_H_TOTAL] + 5;

    hretr_start_char = s->cr[VGA_CRTC_H_SYNC_START];

    hretr_skew_chars = (s->cr[VGA_CRTC_H_SYNC_END] >> 5) & 3;

    hretr_end_char = s->cr[VGA_CRTC_H_SYNC_END] & 0x1f;



    vtotal_lines = (s->cr[VGA_CRTC_V_TOTAL] |

                    (((s->cr[VGA_CRTC_OVERFLOW] & 1) |

                      ((s->cr[VGA_CRTC_OVERFLOW] >> 4) & 2)) << 8)) + 2;

    vretr_start_line = s->cr[VGA_CRTC_V_SYNC_START] |

        ((((s->cr[VGA_CRTC_OVERFLOW] >> 2) & 1) |

          ((s->cr[VGA_CRTC_OVERFLOW] >> 6) & 2)) << 8);

    vretr_end_line = s->cr[VGA_CRTC_V_SYNC_END] & 0xf;



    clocking_mode = (s->sr[VGA_SEQ_CLOCK_MODE] >> 3) & 1;

    clock_sel = (s->msr >> 2) & 3;

    dots = (s->msr & 1) ? 8 : 9;



    chars_per_sec = clk_hz[clock_sel] / dots;



    htotal_chars <<= clocking_mode;



    r->total_chars = vtotal_lines * htotal_chars;

    if (r->freq) {

        r->ticks_per_char = NANOSECONDS_PER_SECOND / (r->total_chars * r->freq);

    } else {

        r->ticks_per_char = NANOSECONDS_PER_SECOND / chars_per_sec;

    }



    r->vstart = vretr_start_line;

    r->vend = r->vstart + vretr_end_line + 1;



    r->hstart = hretr_start_char + hretr_skew_chars;

    r->hend = r->hstart + hretr_end_char + 1;

    r->htotal = htotal_chars;



#if 0

    div2 = (s->cr[VGA_CRTC_MODE] >> 2) & 1;

    sldiv2 = (s->cr[VGA_CRTC_MODE] >> 3) & 1;

    printf (

        "hz=%f\n"

        "htotal = %d\n"

        "hretr_start = %d\n"

        "hretr_skew = %d\n"

        "hretr_end = %d\n"

        "vtotal = %d\n"

        "vretr_start = %d\n"

        "vretr_end = %d\n"

        "div2 = %d sldiv2 = %d\n"

        "clocking_mode = %d\n"

        "clock_sel = %d %d\n"

        "dots = %d\n"

        "ticks/char = %" PRId64 "\n"

        "\n",

        (double) NANOSECONDS_PER_SECOND / (r->ticks_per_char * r->total_chars),

        htotal_chars,

        hretr_start_char,

        hretr_skew_chars,

        hretr_end_char,

        vtotal_lines,

        vretr_start_line,

        vretr_end_line,

        div2, sldiv2,

        clocking_mode,

        clock_sel,

        clk_hz[clock_sel],

        dots,

        r->ticks_per_char

        );

#endif

}
