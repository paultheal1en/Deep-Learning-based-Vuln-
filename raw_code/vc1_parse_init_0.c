static int vc1_parse_init(AVCodecParserContext *s)

{

    VC1ParseContext *vpc = s->priv_data;

    vpc->v.s.slice_context_count = 1;

    return 0;

}
