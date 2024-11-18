void av_frame_unref(AVFrame *frame)
{
    int i;
    wipe_side_data(frame);
    for (i = 0; i < FF_ARRAY_ELEMS(frame->buf); i++)
        av_buffer_unref(&frame->buf[i]);
    for (i = 0; i < frame->nb_extended_buf; i++)
        av_buffer_unref(&frame->extended_buf[i]);
    av_freep(&frame->extended_buf);
    av_dict_free(&frame->metadata);
    av_buffer_unref(&frame->qp_table_buf);
    get_frame_defaults(frame);
}