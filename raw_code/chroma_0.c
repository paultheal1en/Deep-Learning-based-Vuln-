static void chroma(WaveformContext *s, AVFrame *in, AVFrame *out,

                   int component, int intensity, int offset, int column)

{

    const int plane = s->desc->comp[component].plane;

    const int mirror = s->mirror;

    const int c0_linesize = in->linesize[(plane + 1) % s->ncomp];

    const int c1_linesize = in->linesize[(plane + 2) % s->ncomp];

    const int dst_linesize = out->linesize[plane];

    const int max = 255 - intensity;

    const int src_h = in->height;

    const int src_w = in->width;

    int x, y;



    if (column) {

        const int dst_signed_linesize = dst_linesize * (mirror == 1 ? -1 : 1);



        for (x = 0; x < src_w; x++) {

            const uint8_t *c0_data = in->data[(plane + 1) % s->ncomp];

            const uint8_t *c1_data = in->data[(plane + 2) % s->ncomp];

            uint8_t *dst_data = out->data[plane] + offset * dst_linesize;

            uint8_t * const dst_bottom_line = dst_data + dst_linesize * (s->size - 1);

            uint8_t * const dst_line = (mirror ? dst_bottom_line : dst_data);

            uint8_t *dst = dst_line;



            for (y = 0; y < src_h; y++) {

                const int sum = FFABS(c0_data[x] - 128) + FFABS(c1_data[x] - 128);

                uint8_t *target;

                int p;



                for (p = 256 - sum; p < 256 + sum; p++) {

                    target = dst + x + dst_signed_linesize * p;

                    update(target, max, 1);

                }



                c0_data += c0_linesize;

                c1_data += c1_linesize;

                dst_data += dst_linesize;

            }

        }

    } else {

        const uint8_t *c0_data = in->data[(plane + 1) % s->ncomp];

        const uint8_t *c1_data = in->data[(plane + 2) % s->ncomp];

        uint8_t *dst_data = out->data[plane] + offset;



        if (mirror)

            dst_data += s->size - 1;

        for (y = 0; y < src_h; y++) {

            for (x = 0; x < src_w; x++) {

                const int sum = FFABS(c0_data[x] - 128) + FFABS(c1_data[x] - 128);

                uint8_t *target;

                int p;



                for (p = 256 - sum; p < 256 + sum; p++) {

                    if (mirror)

                        target = dst_data - p;

                    else

                        target = dst_data + p;



                    update(target, max, 1);

                }

            }



            c0_data += c0_linesize;

            c1_data += c1_linesize;

            dst_data += dst_linesize;

        }

    }



    envelope(s, out, plane, (plane + 0) % s->ncomp);

}
