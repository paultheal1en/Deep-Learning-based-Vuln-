static inline int get_amv(Mpeg4DecContext *ctx, int n)

{

    MpegEncContext *s = &ctx->m;

    int x, y, mb_v, sum, dx, dy, shift;

    int len     = 1 << (s->f_code + 4);

    const int a = s->sprite_warping_accuracy;



    if (s->workaround_bugs & FF_BUG_AMV)

        len >>= s->quarter_sample;



    if (s->real_sprite_warping_points == 1) {

        if (ctx->divx_version == 500 && ctx->divx_build == 413)

            sum = s->sprite_offset[0][n] / (1 << (a - s->quarter_sample));

        else

            sum = RSHIFT(s->sprite_offset[0][n] << s->quarter_sample, a);

    } else {

        dx    = s->sprite_delta[n][0];

        dy    = s->sprite_delta[n][1];

        shift = ctx->sprite_shift[0];

        if (n)

            dy -= 1 << (shift + a + 1);

        else

            dx -= 1 << (shift + a + 1);

        mb_v = s->sprite_offset[0][n] + dx * s->mb_x * 16 + dy * s->mb_y * 16;



        sum = 0;

        for (y = 0; y < 16; y++) {

            int v;



            v = mb_v + dy * y;

            // FIXME optimize

            for (x = 0; x < 16; x++) {

                sum += v >> shift;

                v   += dx;

            }

        }

        sum = RSHIFT(sum, a + 8 - s->quarter_sample);

    }



    if (sum < -len)

        sum = -len;

    else if (sum >= len)

        sum = len - 1;



    return sum;

}
