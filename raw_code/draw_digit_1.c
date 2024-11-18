static void draw_digit(int digit, uint8_t *dst, unsigned dst_linesize,

                       unsigned segment_width)

{

#define TOP_HBAR        1

#define MID_HBAR        2

#define BOT_HBAR        4

#define LEFT_TOP_VBAR   8

#define LEFT_BOT_VBAR  16

#define RIGHT_TOP_VBAR 32

#define RIGHT_BOT_VBAR 64

    struct {

        int x, y, w, h;

    } segments[] = {

        { 1,  0, 5, 1 }, /* TOP_HBAR */

        { 1,  6, 5, 1 }, /* MID_HBAR */

        { 1, 12, 5, 1 }, /* BOT_HBAR */

        { 0,  1, 1, 5 }, /* LEFT_TOP_VBAR */

        { 0,  7, 1, 5 }, /* LEFT_BOT_VBAR */

        { 6,  1, 1, 5 }, /* RIGHT_TOP_VBAR */

        { 6,  7, 1, 5 }  /* RIGHT_BOT_VBAR */

    };

    static const unsigned char masks[10] = {

        /* 0 */ TOP_HBAR         |BOT_HBAR|LEFT_TOP_VBAR|LEFT_BOT_VBAR|RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 1 */                                                        RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 2 */ TOP_HBAR|MID_HBAR|BOT_HBAR|LEFT_BOT_VBAR                             |RIGHT_TOP_VBAR,

        /* 3 */ TOP_HBAR|MID_HBAR|BOT_HBAR                            |RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 4 */          MID_HBAR         |LEFT_TOP_VBAR              |RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 5 */ TOP_HBAR|BOT_HBAR|MID_HBAR|LEFT_TOP_VBAR                             |RIGHT_BOT_VBAR,

        /* 6 */ TOP_HBAR|BOT_HBAR|MID_HBAR|LEFT_TOP_VBAR|LEFT_BOT_VBAR               |RIGHT_BOT_VBAR,

        /* 7 */ TOP_HBAR                                              |RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 8 */ TOP_HBAR|BOT_HBAR|MID_HBAR|LEFT_TOP_VBAR|LEFT_BOT_VBAR|RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

        /* 9 */ TOP_HBAR|BOT_HBAR|MID_HBAR|LEFT_TOP_VBAR              |RIGHT_TOP_VBAR|RIGHT_BOT_VBAR,

    };

    unsigned mask = masks[digit];

    int i;



    draw_rectangle(0, dst, dst_linesize, segment_width, 0, 0, 8, 13);

    for (i = 0; i < FF_ARRAY_ELEMS(segments); i++)

        if (mask & (1<<i))

            draw_rectangle(255, dst, dst_linesize, segment_width,

                           segments[i].x, segments[i].y, segments[i].w, segments[i].h);

}
