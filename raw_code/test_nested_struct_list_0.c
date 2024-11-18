static void test_nested_struct_list(gconstpointer opaque)

{

    TestArgs *args = (TestArgs *) opaque;

    const SerializeOps *ops = args->ops;

    UserDefNestedList *listp = NULL, *tmp, *tmp_copy, *listp_copy = NULL;

    Error *err = NULL;

    void *serialize_data;

    int i = 0;



    for (i = 0; i < 8; i++) {

        tmp = g_malloc0(sizeof(UserDefNestedList));

        tmp->value = nested_struct_create();

        tmp->next = listp;

        listp = tmp;

    }

    

    ops->serialize(listp, &serialize_data, visit_nested_struct_list, &err);

    ops->deserialize((void **)&listp_copy, serialize_data,

                     visit_nested_struct_list, &err); 



    g_assert(err == NULL);



    tmp = listp;

    tmp_copy = listp_copy;

    while (listp_copy) {

        g_assert(listp);

        nested_struct_compare(listp->value, listp_copy->value);

        listp = listp->next;

        listp_copy = listp_copy->next;

    }



    qapi_free_UserDefNestedList(tmp);

    qapi_free_UserDefNestedList(tmp_copy);



    ops->cleanup(serialize_data);

    g_free(args);

}
