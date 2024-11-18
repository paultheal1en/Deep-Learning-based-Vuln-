static void latm_write_frame_header(AVFormatContext *s, PutBitContext *bs)

{

    LATMContext *ctx = s->priv_data;

    AVCodecContext *avctx = s->streams[0]->codec;

    GetBitContext gb;

    int header_size;



    /* AudioMuxElement */

    put_bits(bs, 1, !!ctx->counter);



    if (!ctx->counter) {

        init_get_bits(&gb, avctx->extradata, avctx->extradata_size * 8);



        /* StreamMuxConfig */

        put_bits(bs, 1, 0); /* audioMuxVersion */

        put_bits(bs, 1, 1); /* allStreamsSameTimeFraming */

        put_bits(bs, 6, 0); /* numSubFrames */

        put_bits(bs, 4, 0); /* numProgram */

        put_bits(bs, 3, 0); /* numLayer */



        /* AudioSpecificConfig */

        if (ctx->object_type == AOT_ALS) {

            header_size = avctx->extradata_size-(ctx->off + 7) >> 3;

            avpriv_copy_bits(bs, &avctx->extradata[ctx->off], header_size);

        } else {

            avpriv_copy_bits(bs, avctx->extradata, ctx->off + 3);



            if (!ctx->channel_conf) {

                avpriv_copy_pce_data(bs, &gb);

            }

        }



        put_bits(bs, 3, 0); /* frameLengthType */

        put_bits(bs, 8, 0xff); /* latmBufferFullness */



        put_bits(bs, 1, 0); /* otherDataPresent */

        put_bits(bs, 1, 0); /* crcCheckPresent */

    }



    ctx->counter++;

    ctx->counter %= ctx->mod;

}
