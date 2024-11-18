static int skip_data_stream_element(AACContext *ac, GetBitContext *gb)

{

    int byte_align = get_bits1(gb);

    int count = get_bits(gb, 8);

    if (count == 255)

        count += get_bits(gb, 8);

    if (byte_align)

        align_get_bits(gb);



    if (get_bits_left(gb) < 8 * count) {

        av_log(ac->avctx, AV_LOG_ERROR, overread_err);

        return -1;

    }

    skip_bits_long(gb, 8 * count);

    return 0;

}
