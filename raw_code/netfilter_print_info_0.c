static void netfilter_print_info(Monitor *mon, NetFilterState *nf)

{

    char *str;

    ObjectProperty *prop;

    ObjectPropertyIterator iter;

    StringOutputVisitor *ov;



    /* generate info str */

    object_property_iter_init(&iter, OBJECT(nf));

    while ((prop = object_property_iter_next(&iter))) {

        if (!strcmp(prop->name, "type")) {

            continue;

        }

        ov = string_output_visitor_new(false);

        object_property_get(OBJECT(nf), string_output_get_visitor(ov),

                            prop->name, NULL);

        str = string_output_get_string(ov);

        visit_free(string_output_get_visitor(ov));

        monitor_printf(mon, ",%s=%s", prop->name, str);

        g_free(str);

    }

    monitor_printf(mon, "\n");

}
