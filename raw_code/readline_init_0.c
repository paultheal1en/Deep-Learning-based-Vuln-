ReadLineState *readline_init(Monitor *mon,

                             ReadLineCompletionFunc *completion_finder)

{

    ReadLineState *rs = g_malloc0(sizeof(*rs));



    rs->hist_entry = -1;

    rs->mon = mon;

    rs->completion_finder = completion_finder;



    return rs;

}
