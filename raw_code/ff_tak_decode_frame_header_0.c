int ff_tak_decode_frame_header(AVCodecContext *avctx, GetBitContext *gb,

                               TAKStreamInfo *ti, int log_level_offset)

{

    if (get_bits(gb, TAK_FRAME_HEADER_SYNC_ID_BITS) != TAK_FRAME_HEADER_SYNC_ID) {

        av_log(avctx, AV_LOG_ERROR + log_level_offset, "missing sync id\n");

        return AVERROR_INVALIDDATA;

    }



    ti->flags     = get_bits(gb, TAK_FRAME_HEADER_FLAGS_BITS);

    ti->frame_num = get_bits(gb, TAK_FRAME_HEADER_NO_BITS);



    if (ti->flags & TAK_FRAME_FLAG_IS_LAST) {

        ti->last_frame_samples = get_bits(gb, TAK_FRAME_HEADER_SAMPLE_COUNT_BITS) + 1;

        skip_bits(gb, 2);

    } else {

        ti->last_frame_samples = 0;

    }



    if (ti->flags & TAK_FRAME_FLAG_HAS_INFO) {

        avpriv_tak_parse_streaminfo(gb, ti);



        if (get_bits(gb, 6))

            skip_bits(gb, 25);

        align_get_bits(gb);

    }



    if (ti->flags & TAK_FRAME_FLAG_HAS_METADATA)

        return AVERROR_INVALIDDATA;



    skip_bits(gb, 24);



    return 0;

}
