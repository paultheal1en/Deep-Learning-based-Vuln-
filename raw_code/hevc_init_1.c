static int hevc_init(AVCodecParserContext *s)

{

    HEVCContext  *h  = &((HEVCParseContext *)s->priv_data)->h;

    h->HEVClc = av_mallocz(sizeof(HEVCLocalContext));



    h->skipped_bytes_pos_size = INT_MAX;



    return 0;

}