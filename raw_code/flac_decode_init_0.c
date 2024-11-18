static av_cold int flac_decode_init(AVCodecContext *avctx)

{

    FLACContext *s = avctx->priv_data;

    s->avctx = avctx;



    avctx->sample_fmt = SAMPLE_FMT_S16;



    if (avctx->extradata_size > 4) {

        /* initialize based on the demuxer-supplied streamdata header */

        if (avctx->extradata_size == FLAC_STREAMINFO_SIZE) {

            ff_flac_parse_streaminfo(avctx, (FLACStreaminfo *)s,

                                     avctx->extradata);

            allocate_buffers(s);

        } else {

            init_get_bits(&s->gb, avctx->extradata, avctx->extradata_size*8);

            metadata_parse(s);

        }

    }



    return 0;

}
