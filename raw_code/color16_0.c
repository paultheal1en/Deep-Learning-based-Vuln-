static void color16(WaveformContext *s, AVFrame *in, AVFrame *out,

                    int component, int intensity, int offset, int column)

{

    const int plane = s->desc->comp[component].plane;

    const int mirror = s->mirror;

    const int limit = s->size - 1;

    const uint16_t *c0_data = (const uint16_t *)in->data[plane + 0];

    const uint16_t *c1_data = (const uint16_t *)in->data[(plane + 1) % s->ncomp];

    const uint16_t *c2_data = (const uint16_t *)in->data[(plane + 2) % s->ncomp];

    const int c0_linesize = in->linesize[ plane + 0 ] / 2;

    const int c1_linesize = in->linesize[(plane + 1) % s->ncomp] / 2;

    const int c2_linesize = in->linesize[(plane + 2) % s->ncomp] / 2;

    const int d0_linesize = out->linesize[ plane + 0 ] / 2;

    const int d1_linesize = out->linesize[(plane + 1) % s->ncomp] / 2;

    const int d2_linesize = out->linesize[(plane + 2) % s->ncomp] / 2;

    const int src_h = in->height;

    const int src_w = in->width;

    int x, y;



    if (s->mode) {

        const int d0_signed_linesize = d0_linesize * (mirror == 1 ? -1 : 1);

        const int d1_signed_linesize = d1_linesize * (mirror == 1 ? -1 : 1);

        const int d2_signed_linesize = d2_linesize * (mirror == 1 ? -1 : 1);

        uint16_t *d0_data = (uint16_t *)out->data[plane] + offset * d0_linesize;

        uint16_t *d1_data = (uint16_t *)out->data[(plane + 1) % s->ncomp] + offset * d1_linesize;

        uint16_t *d2_data = (uint16_t *)out->data[(plane + 2) % s->ncomp] + offset * d2_linesize;

        uint16_t * const d0_bottom_line = d0_data + d0_linesize * (s->size - 1);

        uint16_t * const d0 = (mirror ? d0_bottom_line : d0_data);

        uint16_t * const d1_bottom_line = d1_data + d1_linesize * (s->size - 1);

        uint16_t * const d1 = (mirror ? d1_bottom_line : d1_data);

        uint16_t * const d2_bottom_line = d2_data + d2_linesize * (s->size - 1);

        uint16_t * const d2 = (mirror ? d2_bottom_line : d2_data);



        for (y = 0; y < src_h; y++) {

            for (x = 0; x < src_w; x++) {

                const int c0 = FFMIN(c0_data[x], limit);

                const int c1 = c1_data[x];

                const int c2 = c2_data[x];



                *(d0 + d0_signed_linesize * c0 + x) = c0;

                *(d1 + d1_signed_linesize * c0 + x) = c1;

                *(d2 + d2_signed_linesize * c0 + x) = c2;

            }



            c0_data += c0_linesize;

            c1_data += c1_linesize;

            c2_data += c2_linesize;

            d0_data += d0_linesize;

            d1_data += d1_linesize;

            d2_data += d2_linesize;

        }

    } else {

        uint16_t *d0_data = (uint16_t *)out->data[plane] + offset;

        uint16_t *d1_data = (uint16_t *)out->data[(plane + 1) % s->ncomp] + offset;

        uint16_t *d2_data = (uint16_t *)out->data[(plane + 2) % s->ncomp] + offset;



        if (mirror) {

            d0_data += s->size - 1;

            d1_data += s->size - 1;

            d2_data += s->size - 1;

        }



        for (y = 0; y < src_h; y++) {

            for (x = 0; x < src_w; x++) {

                const int c0 = FFMIN(c0_data[x], limit);

                const int c1 = c1_data[x];

                const int c2 = c2_data[x];



                if (mirror) {

                    *(d0_data - c0) = c0;

                    *(d1_data - c0) = c1;

                    *(d2_data - c0) = c2;

                } else {

                    *(d0_data + c0) = c0;

                    *(d1_data + c0) = c1;

                    *(d2_data + c0) = c2;

                }

            }



            c0_data += c0_linesize;

            c1_data += c1_linesize;

            c2_data += c2_linesize;

            d0_data += d0_linesize;

            d1_data += d1_linesize;

            d2_data += d2_linesize;

        }

    }



    envelope16(s, out, plane, plane);

}
