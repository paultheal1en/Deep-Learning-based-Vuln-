int ff_draw_slice(AVFilterLink *link, int y, int h, int slice_dir)

{

    uint8_t *src[4], *dst[4];

    int i, j, vsub, ret;

    int (*draw_slice)(AVFilterLink *, int, int, int);



    FF_TPRINTF_START(NULL, draw_slice); ff_tlog_link(NULL, link, 0); ff_tlog(NULL, " y:%d h:%d dir:%d\n", y, h, slice_dir);



    /* copy the slice if needed for permission reasons */

    if (link->src_buf) {

        vsub = av_pix_fmt_descriptors[link->format].log2_chroma_h;



        for (i = 0; i < 4; i++) {

            if (link->src_buf->data[i]) {

                src[i] = link->src_buf-> data[i] +

                    (y >> (i==1 || i==2 ? vsub : 0)) * link->src_buf-> linesize[i];

                dst[i] = link->cur_buf->data[i] +

                    (y >> (i==1 || i==2 ? vsub : 0)) * link->cur_buf->linesize[i];

            } else

                src[i] = dst[i] = NULL;

        }



        for (i = 0; i < 4; i++) {

            int planew =

                av_image_get_linesize(link->format, link->cur_buf->video->w, i);



            if (!src[i]) continue;



            for (j = 0; j < h >> (i==1 || i==2 ? vsub : 0); j++) {

                memcpy(dst[i], src[i], planew);

                src[i] += link->src_buf->linesize[i];

                dst[i] += link->cur_buf->linesize[i];

            }

        }

    }



    if (!(draw_slice = link->dstpad->draw_slice))

        draw_slice = default_draw_slice;

    ret = draw_slice(link, y, h, slice_dir);

    if (ret < 0)

        clear_link(link);

    return ret;

}
