static void dealloc_helper(void *native_in, VisitorFunc visit, Error **errp)

{

    QapiDeallocVisitor *qdv = qapi_dealloc_visitor_new();



    visit(qapi_dealloc_get_visitor(qdv), &native_in, errp);



    qapi_dealloc_visitor_cleanup(qdv);

}
