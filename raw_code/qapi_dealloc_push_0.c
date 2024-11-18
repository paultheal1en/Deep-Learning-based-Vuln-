static void qapi_dealloc_push(QapiDeallocVisitor *qov, void *value)

{

    StackEntry *e = g_malloc0(sizeof(*e));



    e->value = value;



    /* see if we're just pushing a list head tracker */

    if (value == NULL) {

        e->is_list_head = true;

    }

    QTAILQ_INSERT_HEAD(&qov->stack, e, node);

}
