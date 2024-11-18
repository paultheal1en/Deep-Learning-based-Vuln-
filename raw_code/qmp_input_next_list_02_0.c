static GenericList *qmp_input_next_list(Visitor *v, GenericList *tail,

                                        size_t size)

{

    QmpInputVisitor *qiv = to_qiv(v);

    StackObject *so = &qiv->stack[qiv->nb_stack - 1];



    if (!so->entry) {

        return NULL;

    }

    tail->next = g_malloc0(size);

    return tail->next;

}
