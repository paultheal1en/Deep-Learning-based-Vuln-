static void blend_subrect(AVPicture *dst, const AVSubtitleRect *rect, int imgw, int imgh)

{

    int x, y, Y, U, V, A;

    uint8_t *lum, *cb, *cr;

    int dstx, dsty, dstw, dsth;

    const AVPicture *src = &rect->pict;



    dstw = av_clip(rect->w, 0, imgw);

    dsth = av_clip(rect->h, 0, imgh);

    dstx = av_clip(rect->x, 0, imgw - dstw);

    dsty = av_clip(rect->y, 0, imgh - dsth);

    lum = dst->data[0] + dstx + dsty * dst->linesize[0];

    cb  = dst->data[1] + dstx/2 + (dsty >> 1) * dst->linesize[1];

    cr  = dst->data[2] + dstx/2 + (dsty >> 1) * dst->linesize[2];



    for (y = 0; y<dsth; y++) {

        for (x = 0; x<dstw; x++) {

            Y = src->data[0][x + y*src->linesize[0]];

            A = src->data[3][x + y*src->linesize[3]];

            lum[0] = ALPHA_BLEND(A, lum[0], Y, 0);

            lum++;

        }

        lum += dst->linesize[0] - dstw;

    }



    for (y = 0; y<dsth/2; y++) {

        for (x = 0; x<dstw/2; x++) {

            U = src->data[1][x + y*src->linesize[1]];

            V = src->data[2][x + y*src->linesize[2]];

            A = src->data[3][2*x     +  2*y   *src->linesize[3]]

              + src->data[3][2*x + 1 +  2*y   *src->linesize[3]]

              + src->data[3][2*x + 1 + (2*y+1)*src->linesize[3]]

              + src->data[3][2*x     + (2*y+1)*src->linesize[3]];

            cb[0] = ALPHA_BLEND(A>>2, cb[0], U, 0);

            cr[0] = ALPHA_BLEND(A>>2, cr[0], V, 0);

            cb++;

            cr++;

        }

        cb += dst->linesize[1] - dstw/2;

        cr += dst->linesize[2] - dstw/2;

    }

}
