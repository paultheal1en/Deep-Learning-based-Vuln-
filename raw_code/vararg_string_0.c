static void vararg_string(void)

{

    int i;

    struct {

        const char *decoded;

    } test_cases[] = {

        { "hello world" },

        { "the quick brown fox jumped over the fence" },

        {}

    };



    for (i = 0; test_cases[i].decoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_jsonf("%s", test_cases[i].decoded);



        g_assert(obj != NULL);

        g_assert(qobject_type(obj) == QTYPE_QSTRING);

        

        str = qobject_to_qstring(obj);

        g_assert(strcmp(qstring_get_str(str), test_cases[i].decoded) == 0);



        QDECREF(str);

    }

}
