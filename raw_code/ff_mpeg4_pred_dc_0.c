static inline int ff_mpeg4_pred_dc(MpegEncContext * s, int n, int level, int *dir_ptr, int encoding)

{

    int a, b, c, wrap, pred, scale, ret;

    int16_t *dc_val;



    /* find prediction */

    if (n < 4) {

        scale = s->y_dc_scale;

    } else {

        scale = s->c_dc_scale;

    }

    if(IS_3IV1)

        scale= 8;



    wrap= s->block_wrap[n];

    dc_val = s->dc_val[0] + s->block_index[n];



    /* B C

     * A X

     */

    a = dc_val[ - 1];

    b = dc_val[ - 1 - wrap];

    c = dc_val[ - wrap];



    /* outside slice handling (we can't do that by memset as we need the dc for error resilience) */

    if(s->first_slice_line && n!=3){

        if(n!=2) b=c= 1024;

        if(n!=1 && s->mb_x == s->resync_mb_x) b=a= 1024;

    }

    if(s->mb_x == s->resync_mb_x && s->mb_y == s->resync_mb_y+1){

        if(n==0 || n==4 || n==5)

            b=1024;

    }



    if (abs(a - b) < abs(b - c)) {

        pred = c;

        *dir_ptr = 1; /* top */

    } else {

        pred = a;

        *dir_ptr = 0; /* left */

    }

    /* we assume pred is positive */

    pred = FASTDIV((pred + (scale >> 1)), scale);



    if(encoding){

        ret = level - pred;

    }else{

        level += pred;

        ret= level;

        if(s->error_recognition>=3){

            if(level<0){

                av_log(s->avctx, AV_LOG_ERROR, "dc<0 at %dx%d\n", s->mb_x, s->mb_y);

                return -1;

            }

            if(level*scale > 2048 + scale){

                av_log(s->avctx, AV_LOG_ERROR, "dc overflow at %dx%d\n", s->mb_x, s->mb_y);

                return -1;

            }

        }

    }

    level *=scale;

    if(level&(~2047)){

        if(level<0)

            level=0;

        else if(!(s->workaround_bugs&FF_BUG_DC_CLIP))

            level=2047;

    }

    dc_val[0]= level;



    return ret;

}
