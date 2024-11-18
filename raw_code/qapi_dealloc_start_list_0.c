static void qapi_dealloc_start_list(Visitor *v, const char *name, Error **errp)

{

    QapiDeallocVisitor *qov = to_qov(v);

    qapi_dealloc_push(qov, NULL);

}
