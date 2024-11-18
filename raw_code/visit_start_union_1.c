bool visit_start_union(Visitor *v, bool data_present, Error **errp)

{

    if (v->start_union) {

        return v->start_union(v, data_present, errp);

    }

    return true;

}
