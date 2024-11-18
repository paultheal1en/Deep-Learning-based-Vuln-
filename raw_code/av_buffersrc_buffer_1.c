int av_buffersrc_buffer(AVFilterContext *ctx, AVFilterBufferRef *buf)

{

    BufferSourceContext *s = ctx->priv;

    AVFrame *frame = NULL;

    AVBufferRef *dummy_buf = NULL;

    int ret = 0, planes, i;



    if (!buf) {

        s->eof = 1;

        return 0;

    } else if (s->eof)

        return AVERROR(EINVAL);



    frame = av_frame_alloc();

    if (!frame)

        return AVERROR(ENOMEM);



    dummy_buf = av_buffer_create(NULL, 0, compat_free_buffer, buf, 0);

    if (!dummy_buf) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    if ((ret = avfilter_copy_buf_props(frame, buf)) < 0)

        goto fail;



#define WRAP_PLANE(ref_out, data, data_size)                            \

do {                                                                    \

    AVBufferRef *dummy_ref = av_buffer_ref(dummy_buf);                  \

    if (!dummy_ref) {                                                   \

        ret = AVERROR(ENOMEM);                                          \

        goto fail;                                                      \

    }                                                                   \

    ref_out = av_buffer_create(data, data_size, compat_unref_buffer,    \

                               dummy_ref, 0);                           \

    if (!ref_out) {                                                     \


        av_frame_unref(frame);                                          \

        ret = AVERROR(ENOMEM);                                          \

        goto fail;                                                      \

    }                                                                   \

} while (0)



    if (ctx->outputs[0]->type  == AVMEDIA_TYPE_VIDEO) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);



        planes = av_pix_fmt_count_planes(frame->format);

        if (!desc || planes <= 0) {

            ret = AVERROR(EINVAL);

            goto fail;

        }



        for (i = 0; i < planes; i++) {

            int v_shift    = (i == 1 || i == 2) ? desc->log2_chroma_h : 0;

            int plane_size = (frame->height >> v_shift) * frame->linesize[i];



            WRAP_PLANE(frame->buf[i], frame->data[i], plane_size);

        }

    } else {

        int planar = av_sample_fmt_is_planar(frame->format);

        int channels = av_get_channel_layout_nb_channels(frame->channel_layout);



        planes = planar ? channels : 1;



        if (planes > FF_ARRAY_ELEMS(frame->buf)) {

            frame->nb_extended_buf = planes - FF_ARRAY_ELEMS(frame->buf);

            frame->extended_buf = av_mallocz(sizeof(*frame->extended_buf) *

                                             frame->nb_extended_buf);

            if (!frame->extended_buf) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

        }



        for (i = 0; i < FFMIN(planes, FF_ARRAY_ELEMS(frame->buf)); i++)

            WRAP_PLANE(frame->buf[i], frame->extended_data[i], frame->linesize[0]);



        for (i = 0; i < planes - FF_ARRAY_ELEMS(frame->buf); i++)

            WRAP_PLANE(frame->extended_buf[i],

                       frame->extended_data[i + FF_ARRAY_ELEMS(frame->buf)],

                       frame->linesize[0]);

    }



    ret = av_buffersrc_add_frame(ctx, frame);



fail:

    av_buffer_unref(&dummy_buf);

    av_frame_free(&frame);



    return ret;

}