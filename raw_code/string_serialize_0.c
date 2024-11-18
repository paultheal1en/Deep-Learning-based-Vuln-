static void string_serialize(void *native_in, void **datap,

                             VisitorFunc visit, Error **errp)

{

    StringSerializeData *d = g_malloc0(sizeof(*d));



    d->sov = string_output_visitor_new(false);

    visit(string_output_get_visitor(d->sov), &native_in, errp);

    *datap = d;

}
