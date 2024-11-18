static int parse_outputs(const char **buf, AVFilterInOut **curr_inputs,

                         AVFilterInOut **open_inputs,

                         AVFilterInOut **open_outputs, AVClass *log_ctx)

{

    int ret, pad = 0;



    while (**buf == '[') {

        char *name = parse_link_name(buf, log_ctx);

        AVFilterInOut *match;



        AVFilterInOut *input = *curr_inputs;







        *curr_inputs = (*curr_inputs)->next;



        if (!name)




        /* First check if the label is not in the open_inputs list */

        match = extract_inout(name, open_inputs);



        if (match) {

            if ((ret = link_filter(input->filter_ctx, input->pad_idx,

                                   match->filter_ctx, match->pad_idx, log_ctx)) < 0)

                return ret;

            av_free(match->name);

            av_free(name);

            av_free(match);

            av_free(input);

        } else {

            /* Not in the list, so add the first input as a open_output */

            input->name = name;

            insert_inout(open_outputs, input);


        *buf += strspn(*buf, WHITESPACES);

        pad++;




    return pad;
