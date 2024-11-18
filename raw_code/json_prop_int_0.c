void json_prop_int(QJSON *json, const char *name, int64_t val)

{

    json_emit_element(json, name);

    qstring_append_int(json->str, val);

}
