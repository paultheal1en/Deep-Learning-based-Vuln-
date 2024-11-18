static void draw_axis_yuv(AVFrame *out, AVFrame *axis, const ColorFloat *c, int off)

{

    int fmt = out->format, x, y, yh, w = axis->width, h = axis->height;

    int offh = (fmt == AV_PIX_FMT_YUV420P) ? off / 2 : off;

    float a, rcp_255 = 1.0f / 255.0f;

    uint8_t *vy = out->data[0], *vu = out->data[1], *vv = out->data[2];

    uint8_t *vay = axis->data[0], *vau = axis->data[1], *vav = axis->data[2], *vaa = axis->data[3];

    int lsy = out->linesize[0], lsu = out->linesize[1], lsv = out->linesize[2];

    int lsay = axis->linesize[0], lsau = axis->linesize[1], lsav = axis->linesize[2], lsaa = axis->linesize[3];

    uint8_t *lpy, *lpu, *lpv, *lpay, *lpau, *lpav, *lpaa;



    for (y = 0; y < h; y += 2) {

        yh = (fmt == AV_PIX_FMT_YUV420P) ? y / 2 : y;

        lpy = vy + (off + y) * lsy;

        lpu = vu + (offh + yh) * lsu;

        lpv = vv + (offh + yh) * lsv;

        lpay = vay + y * lsay;

        lpau = vau + yh * lsau;

        lpav = vav + yh * lsav;

        lpaa = vaa + y * lsaa;

        for (x = 0; x < w; x += 2) {

            a = rcp_255 * (*lpaa++);

            *lpy++ = a * (*lpay++) + (1.0f - a) * c[x].yuv.y + 0.5f;

            *lpu++ = a * (*lpau++) + (1.0f - a) * c[x].yuv.u + 0.5f;

            *lpv++ = a * (*lpav++) + (1.0f - a) * c[x].yuv.v + 0.5f;

            /* u and v are skipped on yuv422p and yuv420p */

            a = rcp_255 * (*lpaa++);

            *lpy++ = a * (*lpay++) + (1.0f - a) * c[x+1].yuv.y + 0.5f;

            if (fmt == AV_PIX_FMT_YUV444P) {

                *lpu++ = a * (*lpau++) + (1.0f - a) * c[x+1].yuv.u + 0.5f;

                *lpv++ = a * (*lpav++) + (1.0f - a) * c[x+1].yuv.v + 0.5f;

            }

        }



        lpy = vy + (off + y + 1) * lsy;

        lpu = vu + (off + y + 1) * lsu;

        lpv = vv + (off + y + 1) * lsv;

        lpay = vay + (y + 1) * lsay;

        lpau = vau + (y + 1) * lsau;

        lpav = vav + (y + 1) * lsav;

        lpaa = vaa + (y + 1) * lsaa;

        for (x = 0; x < out->width; x += 2) {

            /* u and v are skipped on yuv420p */

            a = rcp_255 * (*lpaa++);

            *lpy++ = a * (*lpay++) + (1.0f - a) * c[x].yuv.y + 0.5f;

            if (fmt != AV_PIX_FMT_YUV420P) {

                *lpu++ = a * (*lpau++) + (1.0f - a) * c[x].yuv.u + 0.5f;

                *lpv++ = a * (*lpav++) + (1.0f - a) * c[x].yuv.v + 0.5f;

            }

            /* u and v are skipped on yuv422p and yuv420p */

            a = rcp_255 * (*lpaa++);

            *lpy++ = a * (*lpay++) + (1.0f - a) * c[x+1].yuv.y + 0.5f;

            if (fmt == AV_PIX_FMT_YUV444P) {

                *lpu++ = a * (*lpau++) + (1.0f - a) * c[x+1].yuv.u + 0.5f;

                *lpv++ = a * (*lpav++) + (1.0f - a) * c[x+1].yuv.v + 0.5f;

            }

        }

    }

}
