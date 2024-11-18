int object_property_get_enum(Object *obj, const char *name,

                             const char *strings[], Error **errp)

{

    StringOutputVisitor *sov;

    StringInputVisitor *siv;

    int ret;



    sov = string_output_visitor_new(false);

    object_property_get(obj, string_output_get_visitor(sov), name, errp);

    siv = string_input_visitor_new(string_output_get_string(sov));

    string_output_visitor_cleanup(sov);

    visit_type_enum(string_input_get_visitor(siv),

                    &ret, strings, NULL, name, errp);

    string_input_visitor_cleanup(siv);



    return ret;

}
