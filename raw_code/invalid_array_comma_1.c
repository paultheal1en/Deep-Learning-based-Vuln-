static void invalid_array_comma(void)

{

    QObject *obj = qobject_from_json("[32,}", NULL);

    g_assert(obj == NULL);

}
