void json_prop_str(QJSON *json, const char *name, const char *str)

{

    json_emit_element(json, name);

    qstring_append_chr(json->str, '"');

    qstring_append(json->str, str);

    qstring_append_chr(json->str, '"');

}
