static int compute_mb_distortion(CinepakEncContext *s, AVPicture *a, AVPicture *b)

{

    int x, y, p, d, ret = 0;



    for(y = 0; y < MB_SIZE; y++) {

        for(x = 0; x < MB_SIZE; x++) {

            d = a->data[0][x + y*a->linesize[0]] - b->data[0][x + y*b->linesize[0]];

            ret += d*d;

        }

    }



    if(s->pix_fmt == AV_PIX_FMT_YUV420P) {

        for(p = 1; p <= 2; p++) {

            for(y = 0; y < MB_SIZE/2; y++) {

                for(x = 0; x < MB_SIZE/2; x++) {

                    d = a->data[p][x + y*a->linesize[p]] - b->data[p][x + y*b->linesize[p]];

                    ret += d*d;

                }

            }

        }

    }



    return ret;

}
