static void draw_bar_yuv(AVFrame *out, const float *h, const float *rcp_h,

                         const ColorFloat *c, int bar_h)

{

    int x, y, yh, w = out->width;

    float mul, ht, rcp_bar_h = 1.0f / bar_h;

    uint8_t *vy = out->data[0], *vu = out->data[1], *vv = out->data[2];

    uint8_t *lpy, *lpu, *lpv;

    int lsy = out->linesize[0], lsu = out->linesize[1], lsv = out->linesize[2];

    int fmt = out->format;



    for (y = 0; y < bar_h; y += 2) {

        yh = (fmt == AV_PIX_FMT_YUV420P) ? y / 2 : y;

        ht = (bar_h - y) * rcp_bar_h;

        lpy = vy + y * lsy;

        lpu = vu + yh * lsu;

        lpv = vv + yh * lsv;

        for (x = 0; x < w; x += 2) {

            if (h[x] <= ht) {

                *lpy++ = 16;

                *lpu++ = 128;

                *lpv++ = 128;

            } else {

                mul = (h[x] - ht) * rcp_h[x];

                *lpy++ = mul * c[x].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                *lpu++ = mul * c[x].yuv.u + (1.0f - mul) * 128.0f + 0.5f;

                *lpv++ = mul * c[x].yuv.v + (1.0f - mul) * 128.0f + 0.5f;

            }

            /* u and v are skipped on yuv422p and yuv420p */

            if (fmt == AV_PIX_FMT_YUV444P) {

                if (h[x+1] <= ht) {

                    *lpy++ = 16;

                    *lpu++ = 128;

                    *lpv++ = 128;

                } else {

                    mul = (h[x+1] - ht) * rcp_h[x+1];

                    *lpy++ = mul * c[x+1].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                    *lpu++ = mul * c[x+1].yuv.u + (1.0f - mul) * 128.0f + 0.5f;

                    *lpv++ = mul * c[x+1].yuv.v + (1.0f - mul) * 128.0f + 0.5f;

                }

            } else {

                if (h[x+1] <= ht) {

                    *lpy++ = 16;

                } else {

                    mul = (h[x+1] - ht) * rcp_h[x+1];

                    *lpy++ = mul * c[x+1].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                }

            }

        }



        ht = (bar_h - (y+1)) * rcp_bar_h;

        lpy = vy + (y+1) * lsy;

        lpu = vu + (y+1) * lsu;

        lpv = vv + (y+1) * lsv;

        for (x = 0; x < w; x += 2) {

            /* u and v are skipped on yuv420p */

            if (fmt != AV_PIX_FMT_YUV420P) {

                if (h[x] <= ht) {

                    *lpy++ = 16;

                    *lpu++ = 128;

                    *lpv++ = 128;

                } else {

                    mul = (h[x] - ht) * rcp_h[x];

                    *lpy++ = mul * c[x].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                    *lpu++ = mul * c[x].yuv.u + (1.0f - mul) * 128.0f + 0.5f;

                    *lpv++ = mul * c[x].yuv.v + (1.0f - mul) * 128.0f + 0.5f;

                }

            } else {

                if (h[x] <= ht) {

                    *lpy++ = 16;

                } else {

                    mul = (h[x] - ht) * rcp_h[x];

                    *lpy++ = mul * c[x].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                }

            }

            /* u and v are skipped on yuv422p and yuv420p */

            if (out->format == AV_PIX_FMT_YUV444P) {

                if (h[x+1] <= ht) {

                    *lpy++ = 16;

                    *lpu++ = 128;

                    *lpv++ = 128;

                } else {

                    mul = (h[x+1] - ht) * rcp_h[x+1];

                    *lpy++ = mul * c[x+1].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                    *lpu++ = mul * c[x+1].yuv.u + (1.0f - mul) * 128.0f + 0.5f;

                    *lpv++ = mul * c[x+1].yuv.v + (1.0f - mul) * 128.0f + 0.5f;

                }

            } else {

                if (h[x+1] <= ht) {

                    *lpy++ = 16;

                } else {

                    mul = (h[x+1] - ht) * rcp_h[x+1];

                    *lpy++ = mul * c[x+1].yuv.y + (1.0f - mul) * 16.0f + 0.5f;

                }

            }

        }

    }

}
