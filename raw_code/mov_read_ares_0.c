static int mov_read_ares(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVCodecContext *codec = c->fc->streams[c->fc->nb_streams-1]->codec;

    if (codec->codec_tag == MKTAG('A', 'V', 'i', 'n') &&

        codec->codec_id == AV_CODEC_ID_H264 &&

        atom.size > 11) {

        avio_skip(pb, 10);

        /* For AVID AVCI50, force width of 1440 to be able to select the correct SPS and PPS */

        if (avio_rb16(pb) == 0xd4d)

            codec->width = 1440;

        return 0;

    }



    return mov_read_avid(c, pb, atom);

}
