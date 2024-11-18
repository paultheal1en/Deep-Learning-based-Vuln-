QJSON *qjson_new(void)

{

    QJSON *json = QJSON(object_new(TYPE_QJSON));

    return json;

}
