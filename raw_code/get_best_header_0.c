static int get_best_header(FLACParseContext* fpc, const uint8_t **poutbuf,

                           int *poutbuf_size)

{

    FLACHeaderMarker *header = fpc->best_header;

    FLACHeaderMarker *child  = header->best_child;

    if (!child) {

        *poutbuf_size = av_fifo_size(fpc->fifo_buf) - header->offset;

    } else {

        *poutbuf_size = child->offset - header->offset;



        /* If the child has suspicious changes, log them */

        check_header_mismatch(fpc, header, child, 0);

    }



    if (header->fi.channels != fpc->avctx->channels ||

        (!fpc->avctx->channel_layout && header->fi.channels <= 6)) {

        fpc->avctx->channels = header->fi.channels;

        ff_flac_set_channel_layout(fpc->avctx);

    }

    fpc->avctx->sample_rate = header->fi.samplerate;

    fpc->pc->duration       = header->fi.blocksize;

    *poutbuf = flac_fifo_read_wrap(fpc, header->offset, *poutbuf_size,

                                        &fpc->wrap_buf,

                                        &fpc->wrap_buf_allocated_size);



    fpc->best_header_valid = 0;

    /* Return the negative overread index so the client can compute pos.

       This should be the amount overread to the beginning of the child */

    if (child)

        return child->offset - av_fifo_size(fpc->fifo_buf);

    return 0;

}
