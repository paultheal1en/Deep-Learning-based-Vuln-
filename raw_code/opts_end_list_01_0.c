opts_end_list(Visitor *v, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);



    assert(ov->list_mode == LM_STARTED || ov->list_mode == LM_IN_PROGRESS);

    ov->repeated_opts = NULL;

    ov->list_mode = LM_NONE;

}
