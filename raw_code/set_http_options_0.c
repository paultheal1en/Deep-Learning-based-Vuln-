static void set_http_options(AVFormatContext *s, AVDictionary **options, HLSContext *c)

{

    const char *proto = avio_find_protocol_name(s->filename);

    int http_base_proto = !av_strcasecmp(proto, "http") || !av_strcasecmp(proto, "https");



    if (c->method) {

        av_dict_set(options, "method", c->method, 0);

    } else if (proto && http_base_proto) {

        av_log(c, AV_LOG_WARNING, "No HTTP method set, hls muxer defaulting to method PUT.\n");

        av_dict_set(options, "method", "PUT", 0);

    }

}
