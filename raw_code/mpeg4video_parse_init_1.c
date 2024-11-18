static av_cold int mpeg4video_parse_init(AVCodecParserContext *s)

{

    ParseContext1 *pc = s->priv_data;



    pc->enc = av_mallocz(sizeof(MpegEncContext));

    if (!pc->enc)

        return -1;

    pc->first_picture = 1;


    return 0;

}