static int estimate_sid_gain(G723_1_Context *p)

{

    int i, shift, seg, seg2, t, val, val_add, x, y;



    shift = 16 - p->cur_gain * 2;

    if (shift > 0)

        t = p->sid_gain << shift;

    else

        t = p->sid_gain >> -shift;

    x = av_clipl_int32(t * (int64_t)cng_filt[0] >> 16);



    if (x >= cng_bseg[2])

        return 0x3F;



    if (x >= cng_bseg[1]) {

        shift = 4;

        seg   = 3;

    } else {

        shift = 3;

        seg   = (x >= cng_bseg[0]);

    }

    seg2 = FFMIN(seg, 3);



    val     = 1 << shift;

    val_add = val >> 1;

    for (i = 0; i < shift; i++) {

        t = seg * 32 + (val << seg2);

        t *= t;

        if (x >= t)

            val += val_add;

        else

            val -= val_add;

        val_add >>= 1;

    }



    t = seg * 32 + (val << seg2);

    y = t * t - x;

    if (y <= 0) {

        t = seg * 32 + (val + 1 << seg2);

        t = t * t - x;

        val = (seg2 - 1 << 4) + val;

        if (t >= y)

            val++;

    } else {

        t = seg * 32 + (val - 1 << seg2);

        t = t * t - x;

        val = (seg2 - 1 << 4) + val;

        if (t >= y)

            val--;

    }



    return val;

}
