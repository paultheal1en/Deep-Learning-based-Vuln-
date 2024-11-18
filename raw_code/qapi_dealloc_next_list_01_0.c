static GenericList *qapi_dealloc_next_list(Visitor *v, GenericList **list,

                                           Error **errp)

{

    GenericList *retval = *list;

    g_free(retval->value);

    *list = retval->next;

    return retval;

}
