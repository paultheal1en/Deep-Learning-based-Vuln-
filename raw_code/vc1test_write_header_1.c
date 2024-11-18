static int vc1test_write_header(AVFormatContext *s)

{

    AVCodecContext *avc = s->streams[0]->codec;

    AVIOContext *pb = s->pb;



    if (avc->codec_id != CODEC_ID_WMV3) {

        av_log(s, AV_LOG_ERROR, "Only WMV3 is accepted!\n");

        return -1;

    }

    avio_wl24(pb, 0); //frames count will be here

    avio_w8(pb, 0xC5);

    avio_wl32(pb, 4);

    avio_write(pb, avc->extradata, 4);

    avio_wl32(pb, avc->height);

    avio_wl32(pb, avc->width);

    avio_wl32(pb, 0xC);

    avio_wl24(pb, 0); // hrd_buffer

    avio_w8(pb, 0x80); // level|cbr|res1

    avio_wl32(pb, 0); // hrd_rate

    if (s->streams[0]->r_frame_rate.den && s->streams[0]->r_frame_rate.num == 1)

        avio_wl32(pb, s->streams[0]->r_frame_rate.den);

    else

        avio_wl32(pb, 0xFFFFFFFF); //variable framerate

    avpriv_set_pts_info(s->streams[0], 32, 1, 1000);



    return 0;

}
