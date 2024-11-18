static void blur(CoverContext *cover, AVFrame *in, int offx, int offy)

{

    int x, y, p;



    for (p=0; p<3; p++) {

        int ox = offx>>!!p;

        int oy = offy>>!!p;

        int stride = in->linesize[p];

        uint8_t *data = in->data[p] + ox + oy * stride;

        int w = FF_CEIL_RSHIFT(cover->width , !!p);

        int h = FF_CEIL_RSHIFT(cover->height, !!p);

        int iw = FF_CEIL_RSHIFT(in->width , !!p);

        int ih = FF_CEIL_RSHIFT(in->height, !!p);

        for (y = 0; y < h; y++) {

            for (x = 0; x < w; x++) {

                int c = 0;

                int s = 0;

                if (ox) {

                    int scale = 65536 / (x + 1);

                    s += data[-1 + y*stride] * scale;

                    c += scale;

                }

                if (oy) {

                    int scale = 65536 / (y + 1);

                    s += data[x - stride] * scale;

                    c += scale;

                }

                if (ox + w < iw) {

                    int scale = 65536 / (w - x);

                    s += data[w + y*stride] * scale;

                    c += scale;

                }

                if (oy + h < ih) {

                    int scale = 65536 / (h - y);

                    s += data[x + h*stride] * scale;

                    c += scale;

                }

                data[x + y*stride] = (s + (c>>1)) / c;

            }

        }

    }

}
