opts_end_list(Visitor *v)

{

    OptsVisitor *ov = to_ov(v);



    assert(ov->list_mode == LM_STARTED ||

           ov->list_mode == LM_IN_PROGRESS ||

           ov->list_mode == LM_SIGNED_INTERVAL ||

           ov->list_mode == LM_UNSIGNED_INTERVAL);

    ov->repeated_opts = NULL;

    ov->list_mode = LM_NONE;

}
