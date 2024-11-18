static int decode_iccp_chunk(PNGDecContext *s, int length, AVFrame *f)

{

    int ret, cnt = 0;

    uint8_t *data, profile_name[82];

    AVBPrint bp;

    AVFrameSideData *sd;



    while ((profile_name[cnt++] = bytestream2_get_byte(&s->gb)) && cnt < 81);

    if (cnt > 80) {

        av_log(s->avctx, AV_LOG_ERROR, "iCCP with invalid name!\n");

        return AVERROR_INVALIDDATA;

    }



    length = FFMAX(length - cnt, 0);



    if (bytestream2_get_byte(&s->gb) != 0) {

        av_log(s->avctx, AV_LOG_ERROR, "iCCP with invalid compression!\n");

        return AVERROR_INVALIDDATA;

    }



    length = FFMAX(length - 1, 0);



    if ((ret = decode_zbuf(&bp, s->gb.buffer, s->gb.buffer + length)) < 0)

        return ret;



    av_bprint_finalize(&bp, (char **)&data);





    sd = av_frame_new_side_data(f, AV_FRAME_DATA_ICC_PROFILE, bp.len);

    if (!sd) {

        av_free(data);


    }



    av_dict_set(&sd->metadata, "name", profile_name, 0);

    memcpy(sd->data, data, bp.len);

    av_free(data);



    /* ICC compressed data and CRC */

    bytestream2_skip(&s->gb, length + 4);



    return 0;

}