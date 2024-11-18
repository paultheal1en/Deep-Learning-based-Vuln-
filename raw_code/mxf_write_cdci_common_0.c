static void mxf_write_cdci_common(AVFormatContext *s, AVStream *st, const UID key, unsigned size)

{

    MXFStreamContext *sc = st->priv_data;

    AVIOContext *pb = s->pb;

    int stored_height = (st->codec->height+15)/16*16;

    int display_height;

    int f1, f2;

    unsigned desc_size = size+8+8+8+8+8+8+5+16+sc->interlaced*4+12+20;

    if (sc->interlaced && sc->field_dominance)

        desc_size += 5;



    mxf_write_generic_desc(s, st, key, desc_size);



    mxf_write_local_tag(pb, 4, 0x3203);

    avio_wb32(pb, st->codec->width);



    mxf_write_local_tag(pb, 4, 0x3202);

    avio_wb32(pb, stored_height>>sc->interlaced);



    mxf_write_local_tag(pb, 4, 0x3209);

    avio_wb32(pb, st->codec->width);



    if (st->codec->height == 608) // PAL + VBI

        display_height = 576;

    else if (st->codec->height == 512)  // NTSC + VBI

        display_height = 486;

    else

        display_height = st->codec->height;



    mxf_write_local_tag(pb, 4, 0x3208);

    avio_wb32(pb, display_height>>sc->interlaced);



    // component depth

    mxf_write_local_tag(pb, 4, 0x3301);

    avio_wb32(pb, sc->component_depth);



    // horizontal subsampling

    mxf_write_local_tag(pb, 4, 0x3302);

    avio_wb32(pb, 2);



    // frame layout

    mxf_write_local_tag(pb, 1, 0x320C);

    avio_w8(pb, sc->interlaced);



    // video line map

    switch (st->codec->height) {

    case  576: f1 = 23; f2 = st->codec->codec_id == AV_CODEC_ID_DVVIDEO ? 335 : 336; break;

    case  608: f1 =  7; f2 = 320; break;

    case  480: f1 = 20; f2 = st->codec->codec_id == AV_CODEC_ID_DVVIDEO ? 285 : 283; break;

    case  512: f1 =  7; f2 = 270; break;

    case  720: f1 = 26; f2 =   0; break; // progressive

    case 1080: f1 = 21; f2 = 584; break;

    default:   f1 =  0; f2 =   0; break;

    }



    if (!sc->interlaced) {

        f2  = 0;

        f1 *= 2;

    }



    mxf_write_local_tag(pb, 12+sc->interlaced*4, 0x320D);

    avio_wb32(pb, sc->interlaced ? 2 : 1);

    avio_wb32(pb, 4);

    avio_wb32(pb, f1);

    if (sc->interlaced)

        avio_wb32(pb, f2);



    mxf_write_local_tag(pb, 8, 0x320E);

    avio_wb32(pb, sc->aspect_ratio.num);

    avio_wb32(pb, sc->aspect_ratio.den);



    mxf_write_local_tag(pb, 16, 0x3201);

    avio_write(pb, *sc->codec_ul, 16);



    if (sc->interlaced && sc->field_dominance) {

        mxf_write_local_tag(pb, 1, 0x3212);

        avio_w8(pb, sc->field_dominance);

    }



}
