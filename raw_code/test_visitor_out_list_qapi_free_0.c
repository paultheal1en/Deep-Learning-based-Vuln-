static void test_visitor_out_list_qapi_free(TestOutputVisitorData *data,

                                            const void *unused)

{

    UserDefTwoList *p, *head = NULL;

    const char string[] = "foo bar";

    int i, max_count = 1024;



    for (i = 0; i < max_count; i++) {

        p = g_malloc0(sizeof(*p));

        p->value = g_malloc0(sizeof(*p->value));



        p->value->string0 = g_strdup(string);

        p->value->dict1 = g_new0(UserDefTwoDict, 1);

        p->value->dict1->string1 = g_strdup(string);

        p->value->dict1->dict2 = g_new0(UserDefTwoDictDict, 1);

        p->value->dict1->dict2->userdef = g_new0(UserDefOne, 1);

        p->value->dict1->dict2->userdef->string = g_strdup(string);

        p->value->dict1->dict2->userdef->integer = 42;

        p->value->dict1->dict2->string = g_strdup(string);

        p->value->dict1->has_dict3 = false;



        p->next = head;

        head = p;

    }



    qapi_free_UserDefTwoList(head);

}
