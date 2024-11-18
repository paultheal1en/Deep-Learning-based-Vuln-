static void qapi_dealloc_type_str(Visitor *v, char **obj, const char *name,

                                  Error **errp)

{

    g_free(*obj);

}
