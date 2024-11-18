static int mov_write_uuidprof_tag(AVIOContext *pb, AVFormatContext *s)

{

    AVStream       *video_st    = s->streams[0];

    AVCodecParameters *video_par = s->streams[0]->codecpar;

    AVCodecParameters *audio_par = s->streams[1]->codecpar;

    int audio_rate = audio_par->sample_rate;

    int64_t frame_rate = (video_st->avg_frame_rate.num * 0x10000LL) / video_st->avg_frame_rate.den;

    int audio_kbitrate = audio_par->bit_rate / 1000;

    int video_kbitrate = FFMIN(video_par->bit_rate / 1000, 800 - audio_kbitrate);



    if (frame_rate < 0 || frame_rate > INT32_MAX) {

        av_log(s, AV_LOG_ERROR, "Frame rate %f outside supported range\n", frame_rate / (double)0x10000);

        return AVERROR(EINVAL);

    }



    avio_wb32(pb, 0x94); /* size */

    ffio_wfourcc(pb, "uuid");

    ffio_wfourcc(pb, "PROF");



    avio_wb32(pb, 0x21d24fce); /* 96 bit UUID */

    avio_wb32(pb, 0xbb88695c);

    avio_wb32(pb, 0xfac9c740);



    avio_wb32(pb, 0x0);  /* ? */

    avio_wb32(pb, 0x3);  /* 3 sections ? */



    avio_wb32(pb, 0x14); /* size */

    ffio_wfourcc(pb, "FPRF");

    avio_wb32(pb, 0x0);  /* ? */

    avio_wb32(pb, 0x0);  /* ? */

    avio_wb32(pb, 0x0);  /* ? */



    avio_wb32(pb, 0x2c);  /* size */

    ffio_wfourcc(pb, "APRF"); /* audio */

    avio_wb32(pb, 0x0);

    avio_wb32(pb, 0x2);   /* TrackID */

    ffio_wfourcc(pb, "mp4a");

    avio_wb32(pb, 0x20f);

    avio_wb32(pb, 0x0);

    avio_wb32(pb, audio_kbitrate);

    avio_wb32(pb, audio_kbitrate);

    avio_wb32(pb, audio_rate);

    avio_wb32(pb, audio_par->channels);



    avio_wb32(pb, 0x34);  /* size */

    ffio_wfourcc(pb, "VPRF");   /* video */

    avio_wb32(pb, 0x0);

    avio_wb32(pb, 0x1);    /* TrackID */

    if (video_par->codec_id == AV_CODEC_ID_H264) {

        ffio_wfourcc(pb, "avc1");

        avio_wb16(pb, 0x014D);

        avio_wb16(pb, 0x0015);

    } else {

        ffio_wfourcc(pb, "mp4v");

        avio_wb16(pb, 0x0000);

        avio_wb16(pb, 0x0103);

    }

    avio_wb32(pb, 0x0);

    avio_wb32(pb, video_kbitrate);

    avio_wb32(pb, video_kbitrate);

    avio_wb32(pb, frame_rate);

    avio_wb32(pb, frame_rate);

    avio_wb16(pb, video_par->width);

    avio_wb16(pb, video_par->height);

    avio_wb32(pb, 0x010001); /* ? */



    return 0;

}
