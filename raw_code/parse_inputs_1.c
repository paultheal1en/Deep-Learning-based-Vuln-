static int parse_inputs(const char **buf, AVFilterInOut **curr_inputs,

                        AVFilterInOut **open_outputs, AVClass *log_ctx)

{

    int pad = 0;



    while (**buf == '[') {

        char *name = parse_link_name(buf, log_ctx);

        AVFilterInOut *match;



        if (!name)

            return AVERROR(EINVAL);



        /* First check if the label is not in the open_outputs list */

        match = extract_inout(name, open_outputs);



        if (match) {

            av_free(name);

        } else {

            /* Not in the list, so add it as an input */

            match = av_mallocz(sizeof(AVFilterInOut));

            match->name    = name;

            match->pad_idx = pad;

        }



        insert_inout(curr_inputs, match);



        *buf += strspn(*buf, WHITESPACES);

        pad++;

    }



    return pad;

}
