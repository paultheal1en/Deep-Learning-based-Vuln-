lookup_scalar(const OptsVisitor *ov, const char *name, Error **errp)

{

    if (ov->repeated_opts == NULL) {

        GQueue *list;



        /* the last occurrence of any QemuOpt takes effect when queried by name

         */

        list = lookup_distinct(ov, name, errp);

        return list ? g_queue_peek_tail(list) : NULL;

    }

    return g_queue_peek_head(ov->repeated_opts);

}
