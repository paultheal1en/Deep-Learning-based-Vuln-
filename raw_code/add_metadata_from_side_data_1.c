static int add_metadata_from_side_data(AVCodecContext *avctx, AVFrame *frame)

{

    int size, ret = 0;

    const uint8_t *side_metadata;

    const uint8_t *end;



    side_metadata = av_packet_get_side_data(avctx->pkt,

                                            AV_PKT_DATA_STRINGS_METADATA, &size);

    if (!side_metadata)

        goto end;

    end = side_metadata + size;

    while (side_metadata < end) {

        const uint8_t *key = side_metadata;

        const uint8_t *val = side_metadata + strlen(key) + 1;

        int ret = av_dict_set(avpriv_frame_get_metadatap(frame), key, val, 0);

        if (ret < 0)

            break;

        side_metadata = val + strlen(val) + 1;

    }

end:

    return ret;

}
