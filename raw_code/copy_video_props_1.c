static void copy_video_props(AVFilterBufferRefVideoProps *dst, AVFilterBufferRefVideoProps *src) {

    *dst = *src;

    if (src->qp_table) {

        int qsize = src->qp_table_size;

        dst->qp_table = av_malloc(qsize);

        memcpy(dst->qp_table, src->qp_table, qsize);

    }

}
