static void simple_string(void)

{

    int i;

    struct {

        const char *encoded;

        const char *decoded;

    } test_cases[] = {

        { "\"hello world\"", "hello world" },

        { "\"the quick brown fox jumped over the fence\"",

          "the quick brown fox jumped over the fence" },

        {}

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_json(test_cases[i].encoded, NULL);

        str = qobject_to_qstring(obj);

        g_assert(str);

        g_assert(strcmp(qstring_get_str(str), test_cases[i].decoded) == 0);



        str = qobject_to_json(obj);

        g_assert(strcmp(qstring_get_str(str), test_cases[i].encoded) == 0);



        qobject_decref(obj);

        

        QDECREF(str);

    }

}
