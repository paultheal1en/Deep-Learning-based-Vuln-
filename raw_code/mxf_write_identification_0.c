static void mxf_write_identification(AVFormatContext *s)

{

    MXFContext *mxf = s->priv_data;

    AVIOContext *pb = s->pb;

    const char *company = "Libav";

    const char *product = "OP1a Muxer";

    const char *version;

    int length;



    mxf_write_metadata_key(pb, 0x013000);

    PRINT_KEY(s, "identification key", pb->buf_ptr - 16);



    version = s->streams[0]->codec->flags & CODEC_FLAG_BITEXACT ?

        "0.0.0" : AV_STRINGIFY(LIBAVFORMAT_VERSION);

    length = 84 + (strlen(company)+strlen(product)+strlen(version))*2; // utf-16

    klv_encode_ber_length(pb, length);



    // write uid

    mxf_write_local_tag(pb, 16, 0x3C0A);

    mxf_write_uuid(pb, Identification, 0);

    PRINT_KEY(s, "identification uid", pb->buf_ptr - 16);



    // write generation uid

    mxf_write_local_tag(pb, 16, 0x3C09);

    mxf_write_uuid(pb, Identification, 1);



    mxf_write_local_tag_utf16(pb, 0x3C01, company); // Company Name

    mxf_write_local_tag_utf16(pb, 0x3C02, product); // Product Name

    mxf_write_local_tag_utf16(pb, 0x3C04, version); // Version String



    // write product uid

    mxf_write_local_tag(pb, 16, 0x3C05);

    mxf_write_uuid(pb, Identification, 2);



    // modification date

    mxf_write_local_tag(pb, 8, 0x3C06);

    avio_wb64(pb, mxf->timestamp);

}
