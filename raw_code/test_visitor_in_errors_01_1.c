static void test_visitor_in_errors(TestInputVisitorData *data,
                                   const void *unused)
{
    TestStruct *p = NULL;
    Error *err = NULL;
    Visitor *v;
    strList *q = NULL;
    UserDefTwo *r = NULL;
    WrapAlternate *s = NULL;
    v = visitor_input_test_init(data, "{ 'integer': false, 'boolean': 'foo', "
                                "'string': -42 }");
    visit_type_TestStruct(v, NULL, &p, &err);
    g_assert(!p);
    v = visitor_input_test_init(data, "[ '1', '2', false, '3' ]");
    visit_type_strList(v, NULL, &q, &err);
    assert(!q);
}