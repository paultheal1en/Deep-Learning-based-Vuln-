static int parse_icy(HTTPContext *s, const char *tag, const char *p)
{
    int len = 4 + strlen(p) + strlen(tag);
    int is_first = !s->icy_metadata_headers;
    int ret;
    if (s->icy_metadata_headers)
        len += strlen(s->icy_metadata_headers);
    if ((ret = av_reallocp(&s->icy_metadata_headers, len)) < 0)
        return ret;
    av_strlcatf(s->icy_metadata_headers, len, "%s: %s\n", tag, p);
    return 0;
}