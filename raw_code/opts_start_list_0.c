opts_start_list(Visitor *v, const char *name, Error **errp)

{

    OptsVisitor *ov = to_ov(v);



    /* we can't traverse a list in a list */

    assert(ov->list_mode == LM_NONE);

    ov->repeated_opts = lookup_distinct(ov, name, errp);

    if (ov->repeated_opts != NULL) {

        ov->list_mode = LM_STARTED;

    }

}
