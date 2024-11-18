static void close_slave(TeeSlave *tee_slave)

{

    AVFormatContext *avf;

    unsigned i;



    avf = tee_slave->avf;

    for (i = 0; i < avf->nb_streams; ++i) {

        AVBitStreamFilterContext *bsf_next, *bsf = tee_slave->bsfs[i];

        while (bsf) {

            bsf_next = bsf->next;

            av_bitstream_filter_close(bsf);

            bsf = bsf_next;

        }

    }

    av_freep(&tee_slave->stream_map);

    av_freep(&tee_slave->bsfs);



    ff_format_io_close(avf, &avf->pb);

    avformat_free_context(avf);

    tee_slave->avf = NULL;

}
