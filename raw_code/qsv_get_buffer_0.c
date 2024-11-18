static int qsv_get_buffer(AVCodecContext *s, AVFrame *frame, int flags)

{

    InputStream *ist = s->opaque;

    QSVContext  *qsv = ist->hwaccel_ctx;

    int i;



    for (i = 0; i < qsv->nb_surfaces; i++) {

        if (qsv->surface_used[i])

            continue;



        frame->buf[0] = av_buffer_create((uint8_t*)qsv->surface_ptrs[i], sizeof(*qsv->surface_ptrs[i]),

                                         buffer_release, &qsv->surface_used[i], 0);

        if (!frame->buf[0])

            return AVERROR(ENOMEM);

        frame->data[3]       = (uint8_t*)qsv->surface_ptrs[i];

        qsv->surface_used[i] = 1;

        return 0;

    }



    return AVERROR(ENOMEM);

}
