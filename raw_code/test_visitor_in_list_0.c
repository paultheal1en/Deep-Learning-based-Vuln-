static void test_visitor_in_list(TestInputVisitorData *data,

                                 const void *unused)

{

    UserDefOneList *item, *head = NULL;

    Error *err = NULL;

    Visitor *v;

    int i;



    v = visitor_input_test_init(data, "[ { 'string': 'string0', 'integer': 42 }, { 'string': 'string1', 'integer': 43 }, { 'string': 'string2', 'integer': 44 } ]");



    visit_type_UserDefOneList(v, &head, NULL, &err);

    g_assert(!err);

    g_assert(head != NULL);



    for (i = 0, item = head; item; item = item->next, i++) {

        char string[12];



        snprintf(string, sizeof(string), "string%d", i);

        g_assert_cmpstr(item->value->string, ==, string);

        g_assert_cmpint(item->value->base->integer, ==, 42 + i);

    }



    qapi_free_UserDefOneList(head);

}
