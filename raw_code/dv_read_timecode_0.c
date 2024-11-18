static int dv_read_timecode(AVFormatContext *s) {

    int ret;

    char timecode[AV_TIMECODE_STR_SIZE];

    int64_t pos = avio_tell(s->pb);



    // Read 3 DIF blocks: Header block and 2 Subcode blocks.

    int partial_frame_size = 3 * 80;

    uint8_t *partial_frame = av_mallocz(sizeof(*partial_frame) *

                                        partial_frame_size);



    RawDVContext *c = s->priv_data;

    ret = avio_read(s->pb, partial_frame, partial_frame_size);

    if (ret < 0)

        goto finish;



    if (ret < partial_frame_size) {

        ret = -1;

        goto finish;

    }



    ret = dv_extract_timecode(c->dv_demux, partial_frame, timecode);

    if (ret)

        av_dict_set(&s->metadata, "timecode", timecode, 0);

    else if (ret < 0)

        av_log(s, AV_LOG_ERROR, "Detected timecode is invalid\n");



finish:

    av_free(partial_frame);

    avio_seek(s->pb, pos, SEEK_SET);

    return ret;

}
