static int parse_optional_info(DCACoreDecoder *s)

{

    DCAContext *dca = s->avctx->priv_data;

    int ret = -1;



    // Time code stamp

    if (s->ts_present)

        skip_bits_long(&s->gb, 32);



    // Auxiliary data

    if (s->aux_present && (ret = parse_aux_data(s)) < 0

        && (s->avctx->err_recognition & AV_EF_EXPLODE))

        return ret;



    if (ret < 0)

        s->prim_dmix_embedded = 0;



    // Core extensions

    if (s->ext_audio_present && !dca->core_only) {

        int sync_pos = FFMIN(s->frame_size / 4, s->gb.size_in_bits / 32) - 1;

        int last_pos = get_bits_count(&s->gb) / 32;

        int size, dist;



        // Search for extension sync words aligned on 4-byte boundary. Search

        // must be done backwards from the end of core frame to work around

        // sync word aliasing issues.

        switch (s->ext_audio_type) {

        case EXT_AUDIO_XCH:

            if (dca->request_channel_layout)

                break;



            // The distance between XCH sync word and end of the core frame

            // must be equal to XCH frame size. Off by one error is allowed for

            // compatibility with legacy bitstreams. Minimum XCH frame size is

            // 96 bytes. AMODE and PCHS are further checked to reduce

            // probability of alias sync detection.

            for (; sync_pos >= last_pos; sync_pos--) {

                if (AV_RB32(s->gb.buffer + sync_pos * 4) == DCA_SYNCWORD_XCH) {

                    s->gb.index = (sync_pos + 1) * 32;

                    size = get_bits(&s->gb, 10) + 1;

                    dist = s->frame_size - sync_pos * 4;

                    if (size >= 96

                        && (size == dist || size - 1 == dist)

                        && get_bits(&s->gb, 7) == 0x08) {

                        s->xch_pos = get_bits_count(&s->gb);

                        break;

                    }

                }

            }



            if (s->avctx->err_recognition & AV_EF_EXPLODE) {

                av_log(s->avctx, AV_LOG_ERROR, "XCH sync word not found\n");

                return AVERROR_INVALIDDATA;

            }

            break;



        case EXT_AUDIO_X96:

            // The distance between X96 sync word and end of the core frame

            // must be equal to X96 frame size. Minimum X96 frame size is 96

            // bytes.

            for (; sync_pos >= last_pos; sync_pos--) {

                if (AV_RB32(s->gb.buffer + sync_pos * 4) == DCA_SYNCWORD_X96) {

                    s->gb.index = (sync_pos + 1) * 32;

                    size = get_bits(&s->gb, 12) + 1;

                    dist = s->frame_size - sync_pos * 4;

                    if (size >= 96 && size == dist) {

                        s->x96_pos = get_bits_count(&s->gb);

                        break;

                    }

                }

            }



            if (s->avctx->err_recognition & AV_EF_EXPLODE) {

                av_log(s->avctx, AV_LOG_ERROR, "X96 sync word not found\n");

                return AVERROR_INVALIDDATA;

            }

            break;



        case EXT_AUDIO_XXCH:

            if (dca->request_channel_layout)

                break;



            // XXCH frame header CRC must be valid. Minimum XXCH frame header

            // size is 11 bytes.

            for (; sync_pos >= last_pos; sync_pos--) {

                if (AV_RB32(s->gb.buffer + sync_pos * 4) == DCA_SYNCWORD_XXCH) {

                    s->gb.index = (sync_pos + 1) * 32;

                    size = get_bits(&s->gb, 6) + 1;

                    if (size >= 11 &&

                        !ff_dca_check_crc(&s->gb, (sync_pos + 1) * 32,

                                          sync_pos * 32 + size * 8)) {

                        s->xxch_pos = sync_pos * 32;

                        break;

                    }

                }

            }



            if (s->avctx->err_recognition & AV_EF_EXPLODE) {

                av_log(s->avctx, AV_LOG_ERROR, "XXCH sync word not found\n");

                return AVERROR_INVALIDDATA;

            }

            break;

        }

    }



    return 0;

}
