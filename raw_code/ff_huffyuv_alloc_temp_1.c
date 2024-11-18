av_cold int ff_huffyuv_alloc_temp(HYuvContext *s)

{

    int i;



    if (s->bitstream_bpp<24 || s->version > 2) {

        for (i=0; i<3; i++) {

            s->temp[i]= av_malloc(2*s->width + 16);

            if (!s->temp[i])

                return AVERROR(ENOMEM);

            s->temp16[i] = (uint16_t*)s->temp[i];

        }

    } else {

        s->temp[0]= av_mallocz(4*s->width + 16);

        if (!s->temp[0])

            return AVERROR(ENOMEM);

    }

    return 0;

}
