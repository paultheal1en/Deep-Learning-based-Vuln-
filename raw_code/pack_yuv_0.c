static void pack_yuv(TiffEncoderContext * s, uint8_t * dst, int lnum)

{

    AVFrame *p = &s->picture;

    int i, j, k;

    int w = (s->width - 1) / s->subsampling[0] + 1;

    uint8_t *pu = &p->data[1][lnum / s->subsampling[1] * p->linesize[1]];

    uint8_t *pv = &p->data[2][lnum / s->subsampling[1] * p->linesize[2]];

    for (i = 0; i < w; i++){

        for (j = 0; j < s->subsampling[1]; j++)

            for (k = 0; k < s->subsampling[0]; k++)

                *dst++ = p->data[0][(lnum + j) * p->linesize[0] +

                                    i * s->subsampling[0] + k];

        *dst++ = *pu++;

        *dst++ = *pv++;

    }

}
