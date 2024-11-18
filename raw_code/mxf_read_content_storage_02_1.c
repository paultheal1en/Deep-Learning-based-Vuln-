static int mxf_read_content_storage(void *arg, AVIOContext *pb, int tag, int size, UID uid, int64_t klv_offset)

{

    MXFContext *mxf = arg;

    switch (tag) {

    case 0x1901:




        mxf->packages_count = avio_rb32(pb);

        mxf->packages_refs = av_calloc(mxf->packages_count, sizeof(UID));

        if (!mxf->packages_refs)

            return AVERROR(ENOMEM);

        avio_skip(pb, 4); /* useless size of objects, always 16 according to specs */

        avio_read(pb, (uint8_t *)mxf->packages_refs, mxf->packages_count * sizeof(UID));

        break;

    }

    return 0;

}