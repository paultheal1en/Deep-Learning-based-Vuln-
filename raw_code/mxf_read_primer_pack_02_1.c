static int mxf_read_primer_pack(void *arg, AVIOContext *pb, int tag, int size, UID uid, int64_t klv_offset)

{

    MXFContext *mxf = arg;

    int item_num = avio_rb32(pb);

    int item_len = avio_rb32(pb);



    if (item_len != 18) {

        avpriv_request_sample(pb, "Primer pack item length %d", item_len);

        return AVERROR_PATCHWELCOME;

    }

    if (item_num > 65536) {

        av_log(mxf->fc, AV_LOG_ERROR, "item_num %d is too large\n", item_num);

        return AVERROR_INVALIDDATA;

    }





    mxf->local_tags = av_calloc(item_num, item_len);

    if (!mxf->local_tags)

        return AVERROR(ENOMEM);

    mxf->local_tags_count = item_num;

    avio_read(pb, mxf->local_tags, item_num*item_len);

    return 0;

}