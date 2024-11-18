static AVFilterContext *parse_filter(const char **buf, AVFilterGraph *graph,

                                     int index, AVClass *log_ctx)

{

    char *opts = NULL;

    char *name = consume_string(buf);



    if(**buf == '=') {

        (*buf)++;

        opts = consume_string(buf);

    }



    return create_filter(graph, index, name, opts, log_ctx);

}
