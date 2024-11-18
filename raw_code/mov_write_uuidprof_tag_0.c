static void mov_write_uuidprof_tag(ByteIOContext *pb, AVFormatContext *s)

{

    AVCodecContext *VideoCodec = s->streams[0]->codec;

    AVCodecContext *AudioCodec = s->streams[1]->codec;

    int AudioRate = AudioCodec->sample_rate;

    int FrameRate = ((VideoCodec->time_base.den) * (0x10000))/ (VideoCodec->time_base.num);

    int audio_kbitrate= AudioCodec->bit_rate / 1000;

    int video_kbitrate= FFMIN(VideoCodec->bit_rate / 1000, 800 - audio_kbitrate);



    put_be32(pb, 0x94 ); /* size */

    put_tag(pb, "uuid");

    put_tag(pb, "PROF");



    put_be32(pb, 0x21d24fce ); /* 96 bit UUID */

    put_be32(pb, 0xbb88695c );

    put_be32(pb, 0xfac9c740 );



    put_be32(pb, 0x0 );  /* ? */

    put_be32(pb, 0x3 );  /* 3 sections ? */



    put_be32(pb, 0x14 ); /* size */

    put_tag(pb, "FPRF");

    put_be32(pb, 0x0 );  /* ? */

    put_be32(pb, 0x0 );  /* ? */

    put_be32(pb, 0x0 );  /* ? */



    put_be32(pb, 0x2c );  /* size */

    put_tag(pb, "APRF");   /* audio */

    put_be32(pb, 0x0 );

    put_be32(pb, 0x2 );   /* TrackID */

    put_tag(pb, "mp4a");

    put_be32(pb, 0x20f );

    put_be32(pb, 0x0 );

    put_be32(pb, audio_kbitrate);

    put_be32(pb, audio_kbitrate);

    put_be32(pb, AudioRate );

    put_be32(pb, AudioCodec->channels );



    put_be32(pb, 0x34 );  /* size */

    put_tag(pb, "VPRF");   /* video */

    put_be32(pb, 0x0 );

    put_be32(pb, 0x1 );    /* TrackID */

    put_tag(pb, "mp4v");

    put_be32(pb, 0x103 );

    put_be32(pb, 0x0 );

    put_be32(pb, video_kbitrate);

    put_be32(pb, video_kbitrate);

    put_be32(pb, FrameRate);

    put_be32(pb, FrameRate);

    put_be16(pb, VideoCodec->width);

    put_be16(pb, VideoCodec->height);

    put_be32(pb, 0x010001); /* ? */

}
