static inline void get_limits(MpegEncContext *s, int *range, int *xmin, int *ymin, int *xmax, int *ymax, int f_code)

{

    *range = 8 * (1 << (f_code - 1));

    /* XXX: temporary kludge to avoid overflow for msmpeg4 */

    if (s->out_format == FMT_H263 && !s->h263_msmpeg4)

	*range *= 2;



    if (s->unrestricted_mv) {

        *xmin = -16;

        *ymin = -16;

        if (s->h263_plus)

            *range *= 2;

        if(s->avctx->codec->id!=CODEC_ID_MPEG4){

            *xmax = s->mb_width*16;

            *ymax = s->mb_height*16;

        }else {

            *xmax = s->width;

            *ymax = s->height;

        }

    } else {

        *xmin = 0;

        *ymin = 0;

        *xmax = s->mb_width*16 - 16;

        *ymax = s->mb_height*16 - 16;

    }

}
